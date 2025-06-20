# BoxNav

![Demo of an agent operating in a box environment.](demo.gif)

A simple playground for making an agent navigate around some directed corridors represented as overlapping boxes.

## Getting Started in Unreal Engine

### Dependencies

Unreal Engine is needed for data collection, and you will want to either download our packaged simulation or the Unreal Engine 5 (UE) project from our shared Box folder. You will also need to install [ue5osc](../ue5osc/) using instructions found in its README.

### Note about Command Line Arguments

The commands above showcase some examples as to how the script can be ran. Please look into the `boxsim.py` and `boxnavigator.py` files for details over the arguments that can be passed. You can also run `python boxsim.py --help` to see the available arguments.

### Notes about Unreal Engine

- In our packaged simulation you can add the following lines into the Game.ini file found in either:
  - MacOS: (Hidden folder) `\<Packaged_Game_Name>\Epic\ARCSAssets\Saved\Config\Mac`
  - Windows: `\<Packaged_Game_Name>\ARCSAssets\Saved\Config\Windows`

~~~ini
[/ARCSRobots/SimpleRobotCamera/BP_Pawn_SimpleRobotCamera.BP_Pawn_SimpleRobotCamera_C]
UEPort=UE_PORT
PyPort=PY_PORT
RobotVisible=ROBOT_VISIBLE
~~~

### Coordinate Systems

BoxNav uses `matplotlib` to visualization the environment and agent. `matplotlib` uses a standard Cartesian coordinate system with the x-axis pointing right, the y-axis pointing up, and angles increasing when rotating **counter-clockwise** from +x (0°) to +y (90°).

Unreal Engine uses a left-handed coordinate system with the x-axis pointing left, the y-axis pointing up, and angles increasing when rotating **clockwise** from +x (0°) to +y (90°).

| Top-Down   | BoxNav (RH) | BoxNav (RH) | UE (LH)  | UE (LH)     | 180 - BoxNav |
| ---------- | ----------- | ----------- | -------- | ----------- | ------------ |
|            | 0 to 360    | -180 to 180 | 0 to 360 | -180 to 180 |              |
| West       | 0           | 0           | 180      | 180         | 180          |
| North-West | 45          | 45          | 135      | 135         | 135          |
| North      | 90          | 90          | 90       | 90          | 90           |
| North-East | 135         | 135         | 45       | 45          | 45           |
| East       | 180         | 180         | 0        | 0           | 0            |
| South-East | 225         | -135        | 315      | -45         | -45          |
| South      | 270         | -90         | 270      | -90         | -90          |
| South-West | 315         | -45         | 225      | -135        | -135         |
| West       | 360         | 0           | 180      | -180        | -180         |

In our implementation we:

- Sync the exact position and rotation between BoxNav and UE.
- Use a clockwise rotation system (rotate left is negative).
- Label positive rotations as "right" turns.
- Call `axis.invert_xaxis()` to flip the x-axis in the animation.

TODO:
- add a figure
