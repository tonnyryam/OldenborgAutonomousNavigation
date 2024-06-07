from __future__ import annotations

from math import atan2, degrees, sqrt


def approx_equal(a: float, b: float, threshold: float = 0.0001) -> bool:
    """Compare to floats and return true if they are approximately equal."""
    return abs(a - b) < threshold


class Pt:
    """Defines the x and y values of a point in R^2."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def normalized(self) -> Pt:
        # Epsilon to avoid division by zero
        epsilon = 1e-6
        magnitude = self.magnitude() + epsilon
        return Pt(self.x / magnitude, self.y / magnitude)

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def __mul__(self, scale: float) -> Pt:
        return Pt(self.x * scale, self.y * scale)

    def __sub__(self, other: Pt) -> Pt:
        return Pt(self.x - other.x, self.y - other.y)

    def __add__(self, other: Pt) -> Pt:
        return Pt(self.x + other.x, self.y + other.y)

    def __eq__(self, other: Pt) -> bool:
        # Check if two points are approximately equal
        return approx_equal(self.x, other.x) and approx_equal(self.y, other.y)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    @classmethod
    def scalar_product(cls, A: Pt, B: Pt) -> float:
        return A.x * B.x + A.y * B.y

    @classmethod
    def determinant(cls, A: Pt, B: Pt) -> float:
        return A.x * B.y - A.y * B.x

    @classmethod
    def distance(cls, A: Pt, B: Pt) -> float:
        return sqrt((A.x - B.x) ** 2 + (A.y - B.y) ** 2)

    @classmethod
    def angle_between(cls, A: Pt, B: Pt) -> float:
        return atan2(Pt.determinant(A, B), Pt.scalar_product(A, B))


class Box:
    def __init__(self, lower_left: Pt, upper_left: Pt, upper_right: Pt, target: Pt):
        """Create a arbitrarily rotated box."""
        self.A = lower_left
        self.B = upper_left
        self.C = upper_right
        self.target = target

        self.AB = self.B - self.A
        self.dotAB = Pt.scalar_product(self.AB, self.AB)

        self.BC = self.C - self.B
        self.dotBC = Pt.scalar_product(self.BC, self.BC)

        # Used for drawing
        self.left = min(self.A.x, self.B.x, self.C.x)
        self.lower = min(self.A.y, self.B.y, self.C.y)
        self.upper = max(self.A.y, self.B.y, self.C.y)
        self.right = max(self.A.x, self.B.x, self.C.x)
        self.origin = (self.left, self.lower)
        self.width = Pt.distance(self.B, self.C)
        self.height = Pt.distance(self.A, self.B)
        self.angle_degrees = degrees(atan2(self.B.x - self.A.x, self.B.y - self.A.y))

    def point_is_inside(self, M: Pt) -> bool:
        AM = M - self.A
        BM = M - self.B
        return (0 <= Pt.scalar_product(self.AB, AM) <= self.dotAB) and (
            0 <= Pt.scalar_product(self.BC, BM) <= self.dotBC
        )


def aligned_box(
    left: float, right: float, lower: float, upper: float, target: tuple[float, float]
) -> Box:
    """Create a box aligned with the x and y axes."""
    ll = Pt(left, lower)
    ul = Pt(left, upper)
    ur = Pt(right, upper)
    t = Pt(*target)
    return Box(ll, ul, ur, t)


if __name__ == "__main__":
    # Test from here: https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not

    # Describe box with three points
    A = Pt(5, 0)
    B = Pt(0, 2)
    C = Pt(1, 5)
    ignored_target = Pt(0, 0)
    box = Box(A, B, C, ignored_target)

    AB = B - A
    assert AB == Pt(-5, 2)

    BC = C - B
    assert BC == Pt(1, 3)

    dotAB = Pt.scalar_product(AB, AB)
    assert dotAB == 29

    dotBC = Pt.scalar_product(BC, BC)
    assert dotBC == 10

    # First point to test (this point is inside the box)
    M = Pt(4, 2)

    AM = M - A
    assert AM == Pt(-1, 2)

    BM = M - B
    assert BM == Pt(4, 0)

    dotABAM = Pt.scalar_product(AB, AM)
    assert dotABAM == 9

    dotBCBM = Pt.scalar_product(BC, BM)
    assert dotBCBM == 4

    assert box.point_is_inside(M)

    # Second point to test (this point is outside the box)
    M = Pt(6, 1)

    AM = M - A
    assert AM == Pt(1, 1)

    BM = M - B
    assert BM == Pt(6, -1)

    dotABAM = Pt.scalar_product(AB, AM)
    assert dotABAM == -3

    dotBCBM = Pt.scalar_product(BC, BM)
    assert dotBCBM == 3

    assert not box.point_is_inside(M)

    print("All good.")
