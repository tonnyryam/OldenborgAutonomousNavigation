import pathlib
import platform
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from time import sleep
import math
import boxenv

import wandb
from fastai.callback.wandb import WandbCallback
from fastai.vision.learner import load_learner
from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)

from ue5osc import Communicator


@contextmanager
def set_posix_windows():
    # NOTE: This is a workaround for a bug in fastai/pathlib on Windows
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run inference results.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")
    arg_parser.add_argument("wandb_model", help="Path to the model to evaluate.")

    arg_parser.add_argument("output_dir", help="Directory to store saved images.")
    arg_parser.add_argument(
        "--movement_amount",
        type=float,
        default=120.0,
        help="Movement forward per action.",
    )
    arg_parser.add_argument(
        "--rotation_amount",
        type=float,
        default=10.0,
        help="Rotation per action (in degrees for ue5osc).",
    )
    arg_parser.add_argument(
        "--max_actions",
        type=int,
        default=10,
        help="Maximum number of actions to take.",
    )
    return arg_parser.parse_args()


targets = [
    (4940, 870),
    (4000, 870),
    (4000, 400),
    (255, 400),
    (255, -1850),
    (-825, -1850),
    (-825, 2485),
    (150, 2485),
]


def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


# 710 is the distance between where agent starts and first target
# didn't add coordinates for starting point in targets since it isn't a target
distance_between_targets = [710] + [dist(a, b) for a, b in zip(targets, targets[1:])]


# TODO: use boxenv.py for this functionality
def reach_target(num_hit, coords):
    if coords[0] in range(
        targets[num_hit][0] - 20, targets[num_hit][0] + 20
    ) and coords[1] in range(targets[num_hit][1] - 20, targets[num_hit][1] + 20):
        return True


def main():
    args = parse_args()

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

    with Communicator("127.0.0.1", ue_port=7447, py_port=7001) as ue:
        ue.reset()
        print("Connected to", ue.get_project_name())
        print("Saving images to", output_dir)

        targets_reached = 0
        distance = 0

        previous_action = ""
        for action_step in range(args.max_actions):
            # Save image
            image_filename = f"{output_dir}/{action_step:04}.png"
            ue.save_image(image_filename)
            sleep(0.5)

            # Predict correct action
            action_to_take, action_index, action_probs = model.predict(image_filename)
            action_prob = action_probs[action_index]

            # Prevent cycling actions (e.g., left followed by right)
            if action_to_take == "left" and previous_action == "right":
                action_prob = action_probs.argsort()[1]
                action_to_take = model.dls.vocab[action_probs.argsort()[1]]
            elif action_to_take == "right" and previous_action == "left":
                action_prob = action_probs.argsort()[1]
                action_to_take = model.dls.vocab[action_probs.argsort()[1]]

            # set previous_action
            previous_action = action_to_take

            print(f"Moving {action_to_take} with probabilities {action_prob:.2f}")

            # Take action
            match action_to_take:
                case "forward":
                    ue.move_forward(args.movement_amount)
                case "left":
                    ue.rotate_left(args.rotation_amount)
                case "right":
                    ue.rotate_right(args.rotation_amount)
                case _:
                    raise ValueError(f"Unknown action: {action_to_take}")

            # progress through environment
            # TODO: use boxenv for this functionality

            (x, y, z) = ue.get_location()

            x_coord = x
            y_coord = y

            distance = distance_between_targets[targets_reached] - (
                math.sqrt(
                    (round(x_coord - targets[targets_reached][0]) ** 2)
                    + (round(y_coord - targets[targets_reached][1]) ** 2)
                )
            )

            if reach_target(targets_reached, (x, y)):
                targets_reached += 1
                distance = 0
                for i in range(targets_reached):
                    distance += distance_between_targets[i - 1]
                print("Agent has reached target " + str(targets_reached) + "!")

            print("Progress is at " + str((distance / 14515) * 100) + "%!")


if __name__ == "__main__":
    main()
