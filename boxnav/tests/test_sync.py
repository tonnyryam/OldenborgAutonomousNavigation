from math import radians

from boxnav.box import Pt
from boxnav.boxenv import BoxEnv
from boxnav.boxnavigator import Action, BoxNavigator, Navigator
from boxnav.environments import oldenborg_boxes as boxes

box_env = BoxEnv(boxes)

# TODO: move to constructor
starting_box = boxes[0]
initial_x = starting_box.left + starting_box.width / 2
initial_y = starting_box.lower + 50
initial_position = Pt(initial_x, initial_y)
initial_rotation = radians(90)

resolution = "244x244"
quality = 1
image_ext = "png"
translation_increment = 120.0
rotation_increment = radians(1)
distance_threshold = 75
direction_threshold = radians(12)
randomize_interval = int("inf")
navigator = Navigator.PERFECT
anim_ext = "gif"
ue = True
py_port = 7001
ue_port = 7447
save_images = None

agent = BoxNavigator(
    box_env,
    initial_position,
    initial_rotation,
    distance_threshold,
    direction_threshold,
    translation_increment,
    rotation_increment,
    navigator,
    anim_ext,
    ue,
    py_port,
    ue_port,
    resolution,
    quality,
    save_images,
    image_ext,
    randomize_interval,
)

action_sequence = [
    Action.FORWARD,
    Action.ROTATE_LEFT,
    Action.ROTATE_LEFT,
    Action.ROTATE_LEFT,
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
]

for action in action_sequence:
    match action:
        case Action.FORWARD:
            agent.__action_translate(Action.FORWARD)
        case Action.BACKWARD:
            agent.__action_translate(Action.BACKWARD)
        case Action.ROTATE_LEFT:
            agent.__action_rotate(Action.ROTATE_LEFT)
        case Action.ROTATE_RIGHT:
            agent.__action_rotate(Action.ROTATE_RIGHT)

    print("-" * 64)
    print("Action:", action)

    print("BoxNav position:", agent.position)
    print("Unreal position:", agent.ue.get_location())

    print("BoxNav rotation:", agent.rotation)
    print("Unreal rotation:", agent.ue.get_rotation())


print("Done!")
