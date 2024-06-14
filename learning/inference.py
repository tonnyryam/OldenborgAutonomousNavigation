import pathlib
import platform
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from math import radians
from pathlib import Path

import enlighten

import wandb
from fastai.callback.wandb import WandbCallback
from fastai.vision.learner import load_learner
from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)

from boxnav.box import Pt
from boxnav.boxenv import BoxEnv
from boxnav.boxnavigator import (
    Action,
    BoxNavigator,
    Navigator,
    add_box_navigator_arguments,
)
from boxnav.environments import oldenborg_boxes as boxes


@contextmanager
def set_posix_windows():
    # NOTE: This is a workaround for a bug in fastai/pathlib on Windows
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def check_path(directory: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    # Check if directory is empty
    if len(list(Path(path).iterdir())) != 0:
        raise ValueError(f"Directory {path} is not empty.")


def inference_func(model, previous_action: Action, image_file: str):
    # Predict correct action
    action_to_take, action_index, action_probs = model.predict(image_file)
    action_prob = action_probs[action_index]

    # Translate Action to string
    previous = ""
    match previous_action:
        case Action.ROTATE_LEFT:
            previous = "left"
        case Action.ROTATE_RIGHT:
            previous = "right"
        case Action.FORWARD:
            previous = "forward"

    # Prevent cycling actions (e.g., left followed by right)
    # TODO: need to receive previous action
    if action_to_take == "left" and previous == "right":
        action_prob = action_probs[action_probs.argsort()[1]]
        action_to_take = model.dls.vocab[action_probs.argsort()[1]]

    elif action_to_take == "right" and previous == "left":
        action_prob = action_probs[action_probs.argsort()[1]]
        action_to_take = model.dls.vocab[action_probs.argsort()[1]]

    # print(f"Moving {action_to_take} with probability {action_prob:.2f}")

    # Translate fast.ai action to an Action object
    take_action = Action.NO_ACTION
    match action_to_take:
        case "forward":
            take_action = Action.FORWARD
        case "left":
            take_action = Action.ROTATE_LEFT
        case "right":
            take_action = Action.ROTATE_RIGHT
        case _:
            raise ValueError(f"Unknown action: {action_to_take}")

    return take_action


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run inference results.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")
    arg_parser.add_argument("wandb_model", help="Path to the model to evaluate.")

    arg_parser.add_argument("output_dir", help="Directory to store saved images.")

    arg_parser.add_argument(
        "--max_actions",
        type=int,
        default=10,
        help="Maximum number of actions to take.",
    )

    add_box_navigator_arguments(arg_parser)

    return arg_parser.parse_args()


def main():
    args = parse_args()

    args.navigator = Navigator.VISION
    args.ue = True
    args.image_directory = args.output_dir

    wandb_entity = "arcslaboratory"
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    wandb_notes = args.wandb_notes
    wandb_model = args.wandb_model

    run = wandb.init(
        job_type="inference",
        entity=wandb_entity,
        name=wandb_name,
        project=wandb_project,
        notes=wandb_notes,
    )

    if run is None:
        raise Exception("wandb.init() failed")

    # Download the fastai learner
    artifact = run.use_artifact(f"{wandb_model}:latest", type="model")
    model_dir = artifact.download()
    model_filename = Path(model_dir) / (wandb_model + ".pkl")

    # Load the learner and its model
    # TODO: this doesn't load the "best" model, but the last one
    # We should probably also download the weights and load them manually
    if platform.system() == "Windows":
        with set_posix_windows():
            model = load_learner(model_filename)
    else:
        model = load_learner(model_filename)

    # TODO: temporary fix? (we might remove callback on training side)
    model.remove_cb(WandbCallback)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output directory is empty
    if any(output_dir.iterdir()):
        print("Output directory is not empty. Aborting.")
        return

    if args.output_dir:
        args.ue = True

    if args.output_dir:
        check_path(args.output_dir)

    print("Starting inference.")

    box_env = BoxEnv(boxes)

    starting_box = boxes[0]
    initial_x = starting_box.left + starting_box.width / 2
    initial_y = starting_box.lower + 50
    initial_position = Pt(initial_x, initial_y)
    initial_rotation = radians(90)

    # TODO: use context manager for UE connection?
    agent = BoxNavigator(
        box_env,
        initial_position,
        initial_rotation,
        args,
        vision_callback=partial(inference_func, model),
    )

    pbar_manager = enlighten.get_manager()
    navigation_pbar = pbar_manager.counter(total=100, desc="Completion")

    total_actions_taken, corect_action_taken = 0
    forward_count, rotate_left_count, rotate_right_count = 0

    for _ in range(args.max_actions):
        try:
            executed_action, correct_action = agent.execute_navigator_action()

            total_actions_taken += 1
            corect_action_taken += 1 if executed_action == correct_action else 0

            match executed_action:
                case Action.FORWARD:
                    forward_count += 1
                case Action.ROTATE_LEFT:
                    rotate_left_count += 1
                case Action.ROTATE_RIGHT:
                    rotate_right_count += 1

        except Exception as e:
            print(e)
            break

        if agent.is_stuck():
            print("Agent is stuck.")
            break

        # Navigation progress is based on the percentage of the environment navigated
        navigation_pbar.count = int(agent.get_percent_through_env())
        navigation_pbar.update()

    inference_data = [
        agent.get_percent_through_env,
        total_actions_taken,
        corect_action_taken,
        forward_count,
        rotate_left_count,
        rotate_right_count,
    ]

    agent.ue.close_osc()
    navigation_pbar.close()
    pbar_manager.stop()


if __name__ == "__main__":
    main()
