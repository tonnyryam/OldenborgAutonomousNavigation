from boxsim import boxes
import boxnavigator

# 1- assign values to each box according to its % length of the environment

# starts at 4940, 160

targets = [
    # 710 between start and 1st target
    [4940, 870],
    # 940 between 1st and 2nd
    [4000, 870],
    # 470
    [4000, 400],
    # 3745
    [255, 400],
    # 2250
    [255, -1850],
    # 1080
    [-825, -1850],
    # 4335
    [-825, 2485],
    # 975
    [150, 2485],

    #14,515 total distance for perfect path
]

# 2- accumulate which targets it has reached (if any)

boxnavigator.py
if (
            close_enough(self.position, self.target, self.distance_threshold)
            and len(surrounding_boxes) > 1
        ):
# 3- compute % completion of current box it is in
# ((distance from last target to current target - current distance from target) / distance from last target to current target)
# 4- accumulate which targets it has left (if any)

print(str(boxes))
