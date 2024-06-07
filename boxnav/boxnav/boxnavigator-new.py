from enum import Enum
from math import cos, degrees, sin
from pathlib import Path
from random import choice, random
from typing import Callable

from matplotlib.axes import Axes
from matplotlib.patches import Arrow, Rectangle, Wedge

from ue5osc import Communicator

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
    MODEL_INFERENCE = 3

    def __str__(self) -> str:
        return self.name


class BoxNavigatorBase:
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
        sync_with_ue: bool,
        py_port: int | None,
        ue_port: int | None,
        ue_resolution: str | None,
        ue_quality: int | None,
        image_directory: str | None,
        image_extension: str | None,
        inference_func: Callable[[str], Action] | None = None,
    ) -> None:
        self.env = env
        self.initial_position = position
        self.initial_rotation = rotation
        self.final_target = self.env.boxes[-1].target

        # TODO: find appropriate values for these
        self.target_threshold = target_distance_threshold
        self.target_half_wedge = target_direction_threshold / 2.0  # NOTE: radians(6)
        self.translation_increment = translation_increment
        self.rotation_increment = rotation_increment

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

            case Navigator.MODEL_INFERENCE:
                assert sync_with_ue, "Inference navigator requires sync_with_ue."
                assert inference_func, "Inference navigator requires inference_func."

                self.inference_func = inference_func
                self.__compute_action_navigator = self.__compute_action_model_inference

            case _:
                raise NotImplementedError("Unknown navigator type.")

        # TODO: handle UE stuff
        self.sync_with_ue = sync_with_ue
        if self.sync_with_ue:
            assert py_port, "Syncing with UE requires py_port."
            assert ue_port, "Syncing with UE requires ue_port."
            assert ue_resolution, "Syncing with UE requires image_resolution."
            assert ue_quality, "Syncing with UE requires image_quality."

            self.ue = Communicator("127.0.0.1", ue_port, py_port)

            try:
                ue_project_name = self.ue.get_project_name()
                print(f"Connected to {ue_project_name}.")

            except TimeoutError:
                self.ue.close_osc()
                print("Check if UE packaged game is running.")
                raise SystemExit

            self.ue_resolution = ue_resolution
            self.ue.set_resolution(self.ue_resolution)

            self.ue_quality = ue_quality
            self.ue.set_quality(self.ue_quality)

            if image_directory:
                assert image_extension, "Saving images requires image_extension."

                self.image_directory = Path(image_directory).resolve()
                self.image_directory.mkdir(parents=True, exist_ok=True)

                self.image_extension = image_extension

        self.current_trial_num = 0

        # All other member variables are initialized in reset()
        self.reset()

    def reset(self) -> None:
        self.position = self.initial_position
        self.rotation = self.initial_rotation
        self.target = self.env.boxes[0].target

        self.num_actions_executed = 0
        self.current_trial_num += 1

        # self.stuck = False  # Can only be True in unreal wrapper
        # self.previous_target = self.position
        # self.current_box = self.env.boxes[0]  # Start in the first box
        # self.dominant_direction = self.determine_direction_to_target(self.target)
        # self.anchor_1 = self.rotation_anchor(self.target, self.current_box)[0]
        # self.anchor_2 = self.rotation_anchor(self.target, self.current_box)[1]

        # TODO: handle UE stuff
        if self.sync_with_ue:
            self.ue.reset()
            ...

    def display(self, ax: Axes, scale: float) -> None:
        # Plot agent and agent's heading
        ax.plot(self.position.x, self.position.y, "ro")
        wedge_lo = degrees(self.rotation - self.target_half_wedge)
        wedge_hi = degrees(self.rotation + self.target_half_wedge)
        ax.add_patch(Wedge(self.position.xy(), scale, wedge_lo, wedge_hi, color="red"))

        # Plot target and line to target
        ax.plot(self.target.x, self.target.y, "go")
        dxy = (self.target - self.position).normalized() * scale
        ax.add_patch(Arrow(self.position.x, self.position.y, dxy.x, dxy.y, color="g"))

        # # Check if the environment is of type TeleportingNavigator
        # if isinstance(self, TeleportingNavigator):
        #     self.draw_current_past_rectangle(ax, scale)  # Draw the rectangle
        #     ax.plot(self.anchor_1.x, self.anchor_1.y, "mx")
        #     ax.plot(self.anchor_2.x, self.anchor_2.y, "mx")

    def at_final_target(self) -> bool:
        return Pt.distance(self.position, self.final_target) < self.target_threshold

    def is_stuck(self) -> bool:
        pass

    def execute_action(self) -> tuple[Action, Action]:
        # TODO: throw exception if at final target?
        self.__update_target_if_necessary()

        action_correct = self.__compute_action_correct()
        action_navigator = self.__compute_action_navigator()

        match action_navigator:
            case Action.FORWARD:
                action_executed = self.__action_move_forward()

            case Action.BACKWARD:
                action_executed = self.__action_move_backward()

            case Action.ROTATE_LEFT:
                action_executed = self.__action_rotate_left()

            case Action.ROTATE_RIGHT:
                action_executed = self.__action_rotate_right()

            case Action.TELEPORT:
                action_executed = self.__action_teleport()

            case _:
                raise NotImplementedError("Unknown action.")

        # TODO: update being stuck
        # If action is forward or backward and we don't move
        if action_executed == Action.NO_ACTION:
            ...

        self.num_actions_executed += 1

        return action_navigator, action_correct

    def __action_move(self, sign: int) -> Action:
        new_x = self.position.x + sign * self.translation_increment * cos(self.rotation)
        new_y = self.position.y + sign * self.translation_increment * sin(self.rotation)
        possible_new_position = Pt(new_x, new_y)

        # TODO: checks all boxes (can probably make more efficient)
        if self.env.get_boxes_enclosing_point(possible_new_position):
            self.position = possible_new_position

            if self.sync_with_ue:
                self.ue.set_location_xy(self.position.x, self.position.y)

            return Action.FORWARD if sign > 0 else Action.BACKWARD

        else:
            return Action.NO_ACTION

    def __action_move_forward(self) -> Action:
        return self.__action_move(+1)

    def __action_move_backward(self) -> Action:
        return self.__action_move(-1)

    def __action_rotate_left(self) -> Action:
        self.rotation += self.rotation_increment

        if self.sync_with_ue:
            self.ue.set_yaw(self.rotation)

        return Action.ROTATE_LEFT

    def __action_rotate_right(self) -> Action:
        self.rotation -= self.rotation_increment

        if self.sync_with_ue:
            self.ue.set_yaw(self.rotation)

        return Action.ROTATE_RIGHT

    def __action_teleport(self) -> Action: ...

    def __compute_action_correct(self) -> Action:
        # Compute angle between heading and target
        heading_vector = Pt(cos(self.rotation), sin(self.rotation)).normalized()
        target_vector = (self.target - self.position).normalized()
        signed_angle_to_target = Pt.angle_between(heading_vector, target_vector)

        # Already facing correct direction
        if abs(signed_angle_to_target) < self.target_half_wedge:
            action = Action.FORWARD

        # Need to rotate left (think of unit circle); rotation indicated by positive degrees
        elif signed_angle_to_target > 0:
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

    def __compute_action_model_inference(self) -> Action:
        return self.inference_func(self.latest_image_filepath)

    def __update_target_if_necessary(self) -> None:
        surrounding_boxes = self.env.get_boxes_enclosing_point(self.position)
        distance_to_target = Pt.distance(self.position, self.target)

        if distance_to_target < self.target_threshold:
            # This will throw an exception if the boxes do not properly overlap
            self.target = surrounding_boxes[1].target

            # self.current_box = surrounding_boxes[-1]  # Update current box
            # self.dominant_direction = self.determine_direction_to_target(self.target)
            # self.anchor_1 = self.rotation_anchor(self.target, self.current_box)[0]
            # self.anchor_2 = self.rotation_anchor(self.target, self.current_box)[1]
