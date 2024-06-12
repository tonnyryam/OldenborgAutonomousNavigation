from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .box import Box, Pt


class BoxEnv:
    """A simple 2D environment of interconnected boxes.

    The intended set of boxes works like this:
    1. Each box has a target.
    2. The target should be in an overlapping region between two consecutive boxes.
    3. The final box includes the final target of the entire environment.
    """

    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

        # TODO: plotting doesn't work with rotated boxes
        padding = 100
        min_x = min(min(b.A.x, b.B.x, b.C.x) for b in boxes)
        max_x = max(max(b.A.x, b.B.x, b.C.x) for b in boxes)
        min_y = min(min(b.A.y, b.B.y, b.C.y) for b in boxes)
        max_y = max(max(b.A.y, b.B.y, b.C.y) for b in boxes)

        self.xlim = (min_x - padding, max_x + padding)
        self.ylim = (min_y - padding, max_y + padding)

    def display(self, ax: Axes) -> None:
        for box in self.boxes:
            ax.add_patch(
                Rectangle(
                    box.origin,
                    box.width,
                    box.height,
                    angle=box.angle_degrees,
                    facecolor="blue",
                    alpha=0.5,
                )
            )

            ax.plot(box.target.x, box.target.y, "bx")
        # ax.set_xlim(self.xlim[::-1])
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal")

    def get_boxes_enclosing_point(self, pt: Pt) -> list[Box]:
        return [box for box in self.boxes if box.point_is_inside(pt)]
