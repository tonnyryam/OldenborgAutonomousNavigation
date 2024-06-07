from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from math import radians
from pathlib import Path

from tqdm import tqdm

from boxnav.box import Pt
from boxnav.boxenv import BoxEnv
from boxnav.boxnavigator import BoxNavigator, Navigator
from boxnav.environments import oldenborg_boxes as boxes


def check_path(directory: str) -> None:
    path = Path(directory)
    # Create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    # Check if directory is empty
    if len(list(Path(path).iterdir())) != 0:
        raise ValueError(f"Directory {path} is not empty.")


def simulate(args: Namespace) -> None:
    """Create and update the box environment and run the navigator."""

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
        args.distance_threshold,
        args.direction_threshold,
        args.translation_increment,
        args.rotation_increment,
        args.navigator,
        args.anim_ext,
        args.ue,
        args.py_port,
        args.ue_port,
        args.resolution,
        args.quality,
        args.save_images,
        args.image_ext,
    )

    """
    # TODO: for AJC to try out later
    import enlighten

    manager = enlighten.get_manager()
    trial_bar = manager.counter(total=32, desc="Trial")

    for trial in range(32):
        step_bar = manager.counter(total=128, desc="Step", leave=False)
        for step in range(128):
            time.sleep(0.01)
            step_bar.update()
        step_bar.close()
        print("Trial happened")
        trial_bar.update()

    manager.stop()
    """

    for _ in tqdm(range(args.max_total_actions), desc="Actions taken"):
        if agent.is_stuck() or agent.at_final_target():
            # num_actions = agent.num_actions_taken()
            # if agent.at_final_target():
            #     print(f"Agent reached final target in {num_actions} actions.")
            # else:
            #     print(f"Agent was unable to reach final target within {num_actions} actions.")
            agent.reset()
            if args.stop_after_one_trial:
                break

        _ = agent.execute_next_action()

        # if args.ue:
        #     # TODO: turn "20" into a command line argument
        #     if agent.num_actions_taken() % 20 == 0 and args.randomize:
        #         random_surface = random.choice(list(TexturedSurface))
        #         agent.ue.set_texture(random_surface, randrange(42))

    if args.ue:
        agent.ue.close_osc()

    print("Simulation complete.", end=" ")

    if args.anim_ext:
        output_filename = None
        num = 1
        while not output_filename or Path(output_filename).exists():
            output_filename = f"output_{num}.{args.anim_ext}"
            num += 1
        agent.save_animation(output_filename)
        print(f"Animation saved to {output_filename}.")


def main():
    """Parse arguments and run simulation."""

    argparser = ArgumentParser("Navigate around a box environment.")

    #
    # Required arguments
    #

    argparser.add_argument(
        "navigator",
        type=Navigator.argparse,
        choices=list(Navigator),
        help="Navigator to run.",
    )

    #
    # Optional arguments
    #

    argparser.add_argument("--anim_ext", type=str, help="Output format for animation.")

    argparser.add_argument(
        "--save_images",
        type=str,
        help="Directory in which images should be saved (no images saved otherwise).",
    )

    argparser.add_argument(
        "--ue", action="store_true", help="Connect and send command to Unreal Engine."
    )

    argparser.add_argument(
        "--py_port", type=int, default=7001, help="Python OSC server port."
    )

    argparser.add_argument(
        "--ue_port", type=int, default=7447, help="Unreal Engine OSC server port."
    )

    argparser.add_argument(
        "--resolution",
        type=str,
        default="244x244",
        help="Set resolution of images as ResXxResY.",
    )

    # TODO: add a check for valid quality values
    argparser.add_argument(
        "--quality",
        type=str,
        default="1",
        help="Set quality of images in the range 1 to 4.",
    )

    argparser.add_argument(
        "--image_ext", type=str, default="png", help="Output format for images"
    )

    argparser.add_argument(
        "--max_total_actions",
        type=int,
        default=10,
        help="Maximum total allowed actions across all trials.",
    )

    argparser.add_argument(
        "--stop_after_one_trial",
        type=bool,
        default=False,
        help="Stop after one time through the environment (for debugging).",
    )

    argparser.add_argument(
        "--translation_increment",
        type=float,
        default=120.0,
        help="Determines how far to move forward each step.",
    )

    argparser.add_argument(
        "--rotation_increment",
        type=float,
        default=radians(10),
        help="Determines how much to rotate by for each step.",
    )

    argparser.add_argument(
        "--distance_threshold",
        type=int,
        default=75,
        help="Determines how close the robot has to be to the target to activate the next one.",
    )

    argparser.add_argument(
        "--direction_threshold",
        type=int,
        default=radians(12),
        help="Determines how close the robot has to be to the target to activate the next one.",
    )

    argparser.add_argument(
        "--randomize",
        type=bool,
        default=False,
        action=BooleanOptionalAction,
        help="Randomizes the texture of the walls, floors, and ceilings.",
    )

    args = argparser.parse_args()

    if args.save_images:
        args.ue = True

    if args.save_images:
        check_path(args.save_images)

    simulate(args)


if __name__ == "__main__":
    main()
