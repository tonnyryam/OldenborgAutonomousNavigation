from enum import Enum
from math import cos, degrees, sin
from pathlib import Path
from random import choice, random, randrange
from typing import Callable
from time import sleep

from celluloid import Camera
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Wedge

from ue5osc import NUM_TEXTURES, Communicator, TexturedSurface

from .box import Pt
from .boxenv import BoxEnv


class Action(Enum):
    NO_ACTION = -1
    FORWARD = 0
    BACKWARD = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    TELEPORT = 4

    def __str__(self) -> str:
        return self.name


class Navigator(Enum):
    PERFECT = 0
    WANDERING = 1
    TELEPORTING = 2
    VISION = 3

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def argparse(value: str) -> str:
        try:
            return Navigator[value.upper()]  # type: ignore
        except KeyError:
            return value


class BoxNavigator:
    def __init__(
        self,
        env: BoxEnv,
        position: Pt,
        rotation: float,
        target_distance_threshold: float,
        target_direction_threshold: float,
        translation_increment: float,
        rotation_increment: float,
        navigator_type: Navigator,
        animation_extension: str | None,
        sync_with_ue: bool,
        py_port: int | None,
        ue_port: int | None,
        ue_resolution: str | None,
        ue_quality: int | None,
        image_directory: str | None,
        image_extension: str | None,
        randomize_interval: int | None,
        vision_callback: Callable[[str], Action] | None = None,
    ) -> None:
        self.env = env
        self.initial_position = position
        self.initial_rotation = rotation
        self.final_target = self.env.boxes[-1].target

        # TODO: find appropriate values for these
        self.target_threshold = target_distance_threshold
        self.target_half_wedge = target_direction_threshold / 2.0
        self.translation_increment = translation_increment
        self.rotation_increment = rotation_increment
        self.is_stuck_threshold = 10

        self.generating_animation = animation_extension is not None
        if self.generating_animation:
            self.animation_extension = animation_extension
            self.animation_scale = 300
            fig, self.axis = plt.subplots()
            self.camera = Camera(fig)

        match navigator_type:
            case Navigator.PERFECT:
                self.__compute_action_navigator = self.__compute_action_correct

            case Navigator.WANDERING:
                self.possible_actions = [
                    Action.FORWARD,
                    Action.ROTATE_LEFT,
                    Action.ROTATE_RIGHT,
                ]
                # TODO: find appropriate value for this
                self.chance_of_random_action = 0.25
                self.__compute_action_navigator = self.__compute_action_wandering

            case Navigator.TELEPORTING:
                self.__compute_action_navigator = self.__compute_action_teleporting

            case Navigator.VISION:
                assert sync_with_ue, "Vision navigator requires sync_with_ue."
                assert vision_callback, "Vision navigator requires vision_callback."

                self.vision_callback = vision_callback
                self.__compute_action_navigator = self.__compute_action_vision

            case _:
                raise NotImplementedError("Unknown navigator type.")

        self.sync_with_ue = sync_with_ue
        if self.sync_with_ue:
            assert py_port, "Syncing with UE requires py_port."
            assert ue_port, "Syncing with UE requires ue_port."
            assert ue_resolution, "Syncing with UE requires image_resolution."
            assert ue_quality, "Syncing with UE requires image_quality."
            assert randomize_interval, "Syncing with UE requires randomize_interval."

            self.ue = Communicator("127.0.0.1", ue_port, py_port)
            self.ue_resolution = ue_resolution
            self.ue_quality = ue_quality
            self.randomize_interval = randomize_interval

            if image_directory:
                assert image_extension, "Saving images requires image_extension."

                self.image_directory = Path(image_directory).resolve()
                self.image_directory.mkdir(parents=True, exist_ok=True)

                self.image_extension = image_extension
                self.images_saved = 0

        self.num_resets = 0
        self.trial_num = 0

        # All other member variables are initialized in reset()
        self.reset()

    def reset(self) -> None:
        self.position = self.initial_position
        self.rotation = self.initial_rotation
        self.target = self.env.boxes[0].target

        self.num_actions_executed = 0
        self.num_resets += 1
        self.trial_num += 1

        self.is_stuck_counter = 0

        # self.stuck = False  # Can only be True in unreal wrapper
        # self.previous_target = self.position
        # self.current_box = self.env.boxes[0]  # Start in the first box
        # self.dominant_direction = self.determine_direction_to_target(self.target)
        # self.anchor_1 = self.rotation_anchor(self.target, self.current_box)[0]
        # self.anchor_2 = self.rotation_anchor(self.target, self.current_box)[1]

        if self.sync_with_ue:
            try:
                self.ue.set_resolution(self.ue_resolution)
                self.ue.set_quality(self.ue_quality)
                self.ue.set_raycast_length(self.translation_increment)

            except TimeoutError:
                self.ue.close_osc()
                print("Check if UE packaged game is running.")
                raise SystemExit

            print("initial sync: ")
            self.__sync_ue_position()
            self.__sync_ue_rotation()

    def display(self) -> None:
        # Plot agent as a red circle
        self.axis.plot(self.position.x, self.position.y, "ro")

        # Plot agent's heading as a red wedge
        wedge_lo = degrees(self.rotation - self.target_half_wedge)
        wedge_hi = degrees(self.rotation + self.target_half_wedge)
        position = self.position.xy()
        wedge = Wedge(position, self.animation_scale, wedge_lo, wedge_hi, color="red")
        self.axis.add_patch(wedge)

        # Plot target as a green circle
        self.axis.plot(self.target.x, self.target.y, "go")

        # Plot line to target as a green arrow
        dxy = (self.target - self.position).normalized() * self.animation_scale
        arrow = Arrow(self.position.x, self.position.y, dxy.x, dxy.y, color="g")
        self.axis.add_patch(arrow)

        # # Check if the environment is of type TeleportingNavigator
        # if isinstance(self, TeleportingNavigator):
        #     self.draw_current_past_rectangle(ax, scale)  # Draw the rectangle
        #     ax.plot(self.anchor_1.x, self.anchor_1.y, "mx")
        #     ax.plot(self.anchor_2.x, self.anchor_2.y, "mx")

    def save_animation(self, filename: str, progress_bar_callback) -> None:
        animation = self.camera.animate()
        animation.save(filename, progress_callback=progress_bar_callback)

    def at_final_target(self) -> bool:
        return Pt.distance(self.position, self.final_target) < self.target_threshold

    def is_stuck(self) -> bool:
        return self.is_stuck_counter >= self.is_stuck_threshold

    def execute_next_action(self) -> tuple[Action, Action]:
        self.__update_target_if_necessary()

        # Compute the correct action
        action_correct = self.__compute_action_correct()

        if self.sync_with_ue:
            # Save an image if applicable (using correct action's angle to target in file name)
            if self.image_directory:
                # Generate the next filename - Negative because unreal using a left-hand coordinate system
                angle = f"{-self.signed_angle_to_target:+.2f}".replace(".", "p")
                self.latest_image_filepath = (
                    f"{self.image_directory}/"
                    f"{self.trial_num:03}_{self.images_saved:06}_{angle}.{str(self.image_extension).lower()}"
                )

                sleep(0.25)
                self.ue.save_image(self.latest_image_filepath)
                sleep(0.25)

                self.images_saved += 1

            if (
                self.num_actions_executed % self.randomize_interval == 0
                and self.num_actions_executed != 0
            ):
                random_surface = choice(list(TexturedSurface))
                self.ue.set_texture(random_surface, randrange(NUM_TEXTURES))

        # Loop until we have executed an action or til stuck a certain number of times
        while True:
            # If stuck enough times, early return no action
            if self.is_stuck_counter >= self.is_stuck_threshold:
                return Action.NO_ACTION, action_correct

            # Compute the navigator action
            action_navigator = self.__compute_action_navigator()

            # Check if navigator action is possible
            if action_navigator in [Action.FORWARD, Action.BACKWARD]:
                if self.__move_is_possible(action_navigator):
                    self.is_stuck_counter = 0
                    break
                else:
                    self.is_stuck_counter += 1
            else:
                break

        print(
            "\tValid Action - current position, agent: ",
            self.position,
            degrees(self.rotation),
            ", UE: ",
            self.ue.get_location(),
            self.ue.get_rotation(),
            "\n",
        )
        # print("BoxNavigator received action", action_navigator)
        match action_navigator:
            case Action.FORWARD:
                self.__action_translate(action_navigator)
                print("BoxNavigator should execute forward:")

            case Action.BACKWARD:
                self.__action_translate(action_navigator)

            case Action.ROTATE_LEFT:
                self.__action_rotate(action_navigator)

            case Action.ROTATE_RIGHT:
                self.__action_rotate(action_navigator)

            case Action.TELEPORT:
                self.__action_teleport()

            case _:
                raise NotImplementedError("Unknown action.")

        self.num_actions_executed += 1
        # print("Moved ", action_navigator)
        print(
            "\tAfter, ",
            action_navigator,
            ", agent: ",
            self.position,
            degrees(self.rotation),
            ", UE: ",
            self.ue.get_location(),
            self.ue.get_rotation(),
        )

        # Update the animation
        # TODO: Also call this code in the constructor(?)
        if self.generating_animation:
            self.env.display(self.axis)
            self.display()
            self.axis.invert_xaxis()
            self.camera.snap()

        self.previous_action = action_navigator
        return action_navigator, action_correct

    def __move_is_possible(self, direction: Action) -> bool:
        sign = -1 if direction == Action.BACKWARD else 1

        new_x = self.position.x + sign * self.translation_increment * cos(self.rotation)
        new_y = self.position.y + sign * self.translation_increment * sin(self.rotation)
        possible_new_position = Pt(new_x, new_y)

        print(
            "\nCurrent location, agent: ",
            self.position,
            ", UE: ",
            self.ue.get_location(),
        )
        print("Direction: ", degrees(self.rotation))
        print("\tIn Boxes: ", self.env.get_boxes_enclosing_point(self.position))
        print("Resulting location: ", possible_new_position)
        print("\tIn Boxes: ", self.env.get_boxes_enclosing_point(possible_new_position))

        # TODO: checks all boxes (can probably make more efficient)
        print(
            "move_is_possible is",
            len(self.env.get_boxes_enclosing_point(possible_new_position)) > 0,
            "\n",
        )
        return len(self.env.get_boxes_enclosing_point(possible_new_position)) > 0

    def __sync_ue_position(self) -> None:
        try:
            # self.ue.set_location_xy(self.position.x, self.position.y)

            # NOTE: the following moves the UE agent to match the navigator/python
            # there exists code to sync box position to unreal
            # (in boxunreal called sync_box_position_to_unreal)

            # Get z position from UE
            _, _, unreal_z = self.ue.get_location()

            # Get x, y position from boxsim
            x, y = self.position.xy()
            # print("position y: ", y)
            self.ue.set_location(x, y, unreal_z)
            # print(
            #     "Position synced. Current location, agent: ",
            #     self.position,
            #     ", UE: ",
            #     self.ue.get_location(),
            # )

        except TimeoutError:
            self.ue.close_osc()
            print("Could not sync position with UE.")
            raise SystemExit

    def __sync_ue_rotation(self) -> None:
        try:
            # Conversion from Box to unreal location is (180 - boxYaw) = unrealYaw
            unreal_yaw: float = degrees(180 - self.rotation)
            self.ue.set_yaw(unreal_yaw)
            # self.ue.set_yaw(self.rotation)

        except TimeoutError:
            self.ue.close_osc()
            print("Could not sync rotation with UE.")
            raise SystemExit

    def __action_translate(self, direction: Action) -> None:
        sign = -1 if direction == Action.BACKWARD else 1

        new_x = self.position.x + sign * self.translation_increment * cos(self.rotation)
        new_y = self.position.y + sign * self.translation_increment * sin(self.rotation)
        possible_new_position = Pt(new_x, new_y)

        self.position = possible_new_position

        if self.sync_with_ue:
            self.__sync_ue_position()

    def __action_rotate(self, direction: Action) -> None:
        sign = -1 if direction == Action.ROTATE_RIGHT else 1
        self.rotation += sign * self.rotation_increment

        if self.sync_with_ue:
            self.__sync_ue_rotation()

    def __action_teleport(self) -> Action: ...

    def __compute_action_correct(self) -> Action:
        # Compute angle between heading and target
        heading_vector = Pt(cos(self.rotation), sin(self.rotation)).normalized()
        target_vector = (self.target - self.position).normalized()
        self.signed_angle_to_target = Pt.angle_between(heading_vector, target_vector)

        # Already facing correct direction
        if abs(self.signed_angle_to_target) < self.target_half_wedge:
            action = Action.FORWARD

        # Need to rotate left (think of unit circle); rotation indicated by positive degrees
        elif self.signed_angle_to_target > 0:
            action = Action.ROTATE_LEFT

        # Need to rotate right (think of unit circle); rotation indicated by negative degrees
        else:
            action = Action.ROTATE_RIGHT

        return action

    def __compute_action_wandering(self) -> Action:
        # TODO: we can probably do much better than this
        if random() < self.chance_of_random_action:
            return choice(self.possible_actions)
        else:
            return self.__compute_action_correct()

    def __compute_action_teleporting(self) -> Action:
        return Action.TELEPORT

    def __compute_action_vision(self) -> Action:
        # What do we pass to the callback?
        # - number of resets
        # - number of actions executed
        # - ...
        return self.vision_callback(self.latest_image_filepath)

    def __update_target_if_necessary(self) -> None:
        assert not self.at_final_target(), "Already at final target."

        surrounding_boxes = self.env.get_boxes_enclosing_point(self.position)
        distance_to_target = Pt.distance(self.position, self.target)

        if distance_to_target < self.target_threshold:
            # This will throw an exception if the boxes do not properly overlap
            self.target = surrounding_boxes[1].target

            # self.current_box = surrounding_boxes[-1]  # Update current box
            # self.dominant_direction = self.determine_direction_to_target(self.target)
            # self.anchor_1 = self.rotation_anchor(self.target, self.current_box)[0]
            # self.anchor_2 = self.rotation_anchor(self.target, self.current_box)[1]
