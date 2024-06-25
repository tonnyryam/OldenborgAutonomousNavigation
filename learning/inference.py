import pathlib
import platform
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from math import radians
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time
import numpy as np

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


fastai_to_boxnav = {
    "left": Action.ROTATE_LEFT,
    "right": Action.ROTATE_RIGHT,
    "forward": Action.FORWARD,
}

action_prev = Action.NO_ACTION


def inference_func(model, image_file: str):
    global action_prev

    action_now, action_index, action_probs = model.predict(image_file)

    action_now = fastai_to_boxnav[action_now]

    # Prevent cycling actions (e.g., left followed by right)
    right_left = action_now == Action.ROTATE_LEFT and action_prev == Action.ROTATE_RIGHT
    left_right = action_now == Action.ROTATE_RIGHT and action_prev == Action.ROTATE_LEFT
    if right_left or left_right:
        action_index = action_probs.argsort()[1]
        action_now = fastai_to_boxnav[model.dls.vocab[action_index]]

    # TODO: Maybe log with loguru
    # action_prob = action_probs[action_index]
    # print(f"Moving {action_to_take} with probability {action_prob:.2f}")

    action_prev = action_now
    return action_now


colors = ["b", "g", "r", "c", "m", "y", "k"]
line_styles = ["-", "--", "-.", ":"]


def plot_trial(axis: Axes, x_values, y_values) -> None:
    random_color = random.choice(colors)
    random_line_style = random.choice(line_styles)

    axis.plot(x_values, y_values, color=random_color, linestyle=random_line_style)


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run inference results.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")
    arg_parser.add_argument("wandb_model", help="Path to the model to evaluate.")

    arg_parser.add_argument("output_dir", help="Directory to store saved images.")

    arg_parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of times to run model through environment",
    )
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
    trials_pbar = pbar_manager.counter(total=args.num_trials, desc="Trials: ")

    inference_data = []
    executed_actions, correct_actions = [], []
    all_xs, all_ys = [], []
    execution_times = []

    action_to_confusion = {
        Action.FORWARD: 0,
        Action.ROTATE_RIGHT: 1,
        Action.ROTATE_LEFT: 2,
    }

    # Initialize the box image display
    plot_fig, plot_axis = plt.subplots()
    box_env.display(plot_axis)

    for trial in range(args.num_trials):
        total_actions_taken, correct_action_taken = 0, 0
        forward_count, rotate_left_count, rotate_right_count = 0, 0, 0
        incorrect_left_count, incorrect_right_count = 0, 0

        xs, ys = [], []

        actions_pbar = pbar_manager.counter(total=args.max_actions, desc="Actions: ")
        navigation_pbar = pbar_manager.counter(total=100, desc="Completion: ")

        for _ in range(args.max_actions):
            start_time = time.time()
            try:
                executed_action, correct_action = agent.execute_navigator_action()

            except Exception as e:
                print(e)
                break
            end_time = time.time()
            execution_times.append([end_time - start_time])

            if executed_action != Action.NO_ACTION:
                executed_actions.append(action_to_confusion[executed_action])
                correct_actions.append(action_to_confusion[correct_action])

            total_actions_taken += 1
            correct_action_taken += 1 if executed_action == correct_action else 0
            if (
                executed_action == Action.ROTATE_LEFT
                and correct_action == Action.ROTATE_RIGHT
            ):
                incorrect_left_count += 1
            elif (
                executed_action == Action.ROTATE_RIGHT
                and correct_action == Action.ROTATE_LEFT
            ):
                incorrect_right_count += 1

            match executed_action:
                case Action.FORWARD:
                    forward_count += 1
                case Action.ROTATE_LEFT:
                    rotate_left_count += 1
                case Action.ROTATE_RIGHT:
                    rotate_right_count += 1

            current_x, current_y = agent.position.xy()
            xs.append(current_x)
            ys.append(current_y)

            actions_pbar.update()

            # Navigation progress is based on the percentage of the environment navigated
            navigation_pbar.count = int(agent.get_percent_through_env())
            navigation_pbar.update()

            if agent.get_percent_through_env() >= 99.0:
                print("Agent reached final target.")
                break

            elif agent.is_stuck():
                print("Agent is stuck.")
                plot_axis.plot(current_x, current_y, "ro")
                break

        plot_trial(plot_axis, xs, ys)
        all_xs.append(xs)
        all_ys.append(ys)

        run_data = [
            trial,
            agent.get_percent_through_env(),
            total_actions_taken,
            correct_action_taken,
            forward_count,
            rotate_left_count,
            rotate_right_count,
            incorrect_left_count,
            incorrect_right_count,
            total_actions_taken / agent.get_percent_through_env(),
        ]
        inference_data.append(run_data)

        agent.reset()
        trials_pbar.update()
        actions_pbar.close()
        navigation_pbar.close()

    agent.ue.close_osc()
    trials_pbar.close()
    pbar_manager.stop()

    # ------------------------------- DATA PLOTTING IN WANDB -------------------------------
    plot_fig.savefig(str(args.output_dir) + ".png")
    run.log({"Plotted Paths": wandb.Image((str(args.output_dir) + ".png"))})

    # Distribution of how long it takes to execute actions
    time_table = wandb.Table(data=execution_times, columns=["Time Estimate"])
    fields = {"value": "Time Estimate", "title": "Executed Timer Histogram"}
    histogram = wandb.plot_table(
        vega_spec_name="wandb/histogram_small_bins",
        data_table=time_table,
        fields=fields,
    )
    wandb.log({"Executed Timer Histogram": histogram})

    # Calculate summary statistics for action execution times
    execution_times_array = np.array(execution_times)
    percentiles = np.percentile(execution_times_array, [25, 50, 75])

    summary_stats = [
        ["Mean", np.mean(execution_times_array)],
        ["Standard Deviation", np.std(execution_times_array)],
        ["Median", np.median(execution_times_array)],
        ["Minimum Value", np.min(execution_times_array)],
        ["Maximum Value", np.max(execution_times_array)],
        ["25th Percentile", percentiles[0]],
        ["50th Percentile", percentiles[1]],
        ["75th Percentile", percentiles[2]],
    ]
    run.log(
        {
            "Inference Execute Action Timer Data": wandb.Table(
                columns=["Statistic", "Value"], data=summary_stats
            )
        }
    )

    # Confusion Matrix assessing ALL trials
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                preds=executed_actions,
                y_true=correct_actions,
                class_names=["forward", "right", "left"],
            )
        }
    )

    wandb.log(
        {
            "Consistency": wandb.plot.line_series(
                xs=all_xs,
                ys=all_ys,
                keys=[
                    "Trial " + str(trial_num) for trial_num in range(args.num_trials)
                ],
                title="Tracking where Agent has been",
                xname="x location",
            )
        }
    )

    # Implement new table
    table_cols = [
        "Trial",
        "Percent through Environment",
        "Total Actions Taken",
        "Correct Actions Taken",
        "Forward Action Taken",
        "Rotate Left Action Taken",
        "Rotate Right Action Taken",
        "Incorrect Left Taken",
        "Incorrect Right Taken",
        "# Actions per Percent of Env",
    ]
    inference_data_table = wandb.Table(columns=table_cols, data=inference_data)
    run.log({"Inference Data": inference_data_table})

    # Line plot to assess efficiency of model
    wandb.log(
        {
            "Efficiency": wandb.plot.line(
                inference_data_table,
                "# Actions per Percent of Env",
                "Percent through Environment",
                title="Percent through environment vs. # of Actions per percent of Environment",
            )
        }
    )


if __name__ == "__main__":
    main()
