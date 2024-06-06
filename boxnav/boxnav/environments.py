from .box import aligned_box

oldenborg_boxes = [
    aligned_box(left=4640, right=5240, lower=110, upper=1510, target=(4940, 870)),
    aligned_box(left=3720, right=5240, lower=700, upper=1040, target=(4000, 870)),
    aligned_box(left=3850, right=4120, lower=360, upper=1040, target=(4000, 400)),
    aligned_box(left=110, right=4120, lower=260, upper=540, target=(255, 400)),
    aligned_box(left=150, right=400, lower=-1980, upper=540, target=(255, -1850)),
    aligned_box(left=-1550, right=400, lower=-1980, upper=-1720, target=(-825, -1850)),
    aligned_box(left=-900, right=-700, lower=-1980, upper=3320, target=(-825, 2485)),
    aligned_box(left=-900, right=230, lower=2150, upper=2820, target=(150, 2485)),
]
