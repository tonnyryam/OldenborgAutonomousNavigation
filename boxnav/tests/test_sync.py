from argparse import ArgumentParser
from math import radians
from sys import argv
from time import sleep

from boxnav.box import Pt
from boxnav.boxenv import BoxEnv
from boxnav.boxnavigator import Action, BoxNavigator, add_box_navigator_arguments
from boxnav.environments import oldenborg_boxes as boxes

box_env = BoxEnv(boxes)

# TODO: move to constructor
starting_box = boxes[0]
initial_x = starting_box.left + starting_box.width / 2
initial_y = starting_box.lower + 50
initial_position = Pt(initial_x, initial_y)
initial_rotation = radians(90)

argparser = ArgumentParser("Navigate around a box environment.")

add_box_navigator_arguments(argparser)
argparser.add_argument("delay", type=float, help="Delay between actions.")

argv.insert(1, "PERFECT")
argv.append("--ue")

args = argparser.parse_args()

agent = BoxNavigator(box_env, initial_position, initial_rotation, args)


action_sequence = [
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
    Action.FORWARD,
    Action.ROTATE_LEFT,
    Action.ROTATE_LEFT,
    Action.ROTATE_LEFT,
    Action.FORWARD,
    Action.FORWARD,
]

for action in action_sequence:
    print("-" * 64)
    print("Action:", action)

    if args.delay != float("inf"):
        sleep(args.delay)
    else:
        _ = input("Press Enter to continue... ")

    agent.execute_action(action)

    print("BoxNav position:", agent.position)
    print("Unreal position:", agent.ue.get_location())

    print(f"BoxNav rotation: {agent.rotation:0.03}")
    print(f"Unreal rotation: {radians(agent.ue.get_rotation()[2]):0.03}")

    sleep(0.1)


agent.ue.close_osc()

if args.animation_extension:
    agent.save_animation("test_animation.gif")

print("Done!")
