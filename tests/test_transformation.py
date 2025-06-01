import numpy as np
import pytest

from geometer import (
    Circle,
    Line,
    Point,
    PointCollection,
    Transformation,
    TransformationCollection,
    angle,
    reflection,
    rotation,
    scaling,
    translation,
)
from geometer.exceptions import NoIncidence


class TestTransformation:
    def test_from_points(self) -> None:
        p1 = Point(0, 0)
        p2 = Point(1, 0)
        p3 = Point(0, 1)
        p4 = Point(3, 5)
        l = Line(p1, p3)

        M = Transformation.from_points(
            (p1, p1 + Point(1, 1)),  # type: ignore[arg-type]
            (p2, p2 + Point(1, 1)),  # type: ignore[arg-type]
            (p3, p3 + Point(1, 1)),  # type: ignore[arg-type]
            (p4, p4 + Point(1, 1)),  # type: ignore[arg-type]
        )

        assert M * p3 == Point(1, 2)
        assert (M * l).contains(M * p1)
        assert (M * l).contains(M * p3)

    def test_translation(self) -> None:
        p = Point(0, 1)
        t = translation(0, -1)
        assert t * p == Point(0, 0)

        l = Line(Point(0, 0, 1), Point(1, 0, 1))
        t = translation(0, 0, -1)
        assert t * l == Line(Point(0, 0, 0), Point(1, 0, 0))

    def test_inverse(self) -> None:
        E = Transformation(np.eye(4))
        M = rotation(np.pi, axis=Point(0, 1, 0))
        assert M.inverse() * M == E

    def test_pow(self) -> None:
        t = translation(1, 2)

        assert t**0 == Transformation(np.eye(3))
        assert t**1 == t
        assert t**2 == translation(2, 4)
        assert t**3 == translation(3, 6)
        assert t ** (-2) == translation(-2, -4)

    def test_rotation(self) -> None:
        p = Point(0, 1)
        t = rotation(-np.pi)
        assert t * p == Point(0, -1)

        p = Point(1, 0, 0)
        t = rotation(-np.pi / 2, axis=Point(0, 0, 1))
        assert t * p == Point(0, 1, 0)

        p = Point(-1, 1, 0)
        a = np.pi / 7
        t = rotation(a, axis=Point(1, 1, 2))
        assert np.isclose(angle(p, t * p), a)

    def test_scaling(self) -> None:
        p = Point(1, 1, 2)
        s = scaling(3, -4.5, 5)

        assert s * p == Point(3, -4.5, 10)

    def test_reflection(self) -> None:
        p = Point(-1, 1)
        r = reflection(Line(1, -1, 1))

        assert r * p == Point(0, 0)

    def test_from_points_and_conics(self) -> None:
        c1 = Circle()
        p1 = Point(0, -1)
        p2 = Point(0, 1)
        p3 = Point(1, 0)

        c2 = Circle(Point(0, 2), 2)
        q1 = Point(0, 0)
        q2 = Point(0, 4)
        q3 = Point(2, 2)

        t = Transformation.from_points_and_conics([p1, p2, p3], [q1, q2, q3], c1, c2)

        assert t * p1 == q1
        assert t * p2 == q2
        assert t * p2 == q2
        assert t.apply(c1) == c2

        with pytest.raises(NoIncidence, match=r".*Point\(42, 0\).*"):
            Transformation.from_points_and_conics([Point(42, 0), p2, p3], [q1, q2, q3], c1, c2)
        with pytest.raises(NoIncidence, match=r".*Point\(42, 0\).*"):
            Transformation.from_points_and_conics([p1, p2, p3], [Point(42, 0), q2, q3], c1, c2)
        with pytest.raises(NoIncidence, match=r".*Point\(42, 0\).*"):
            Transformation.from_points_and_conics([p1, Point(42, 0), p3], [q1, q2, q3], c1, c2)
        with pytest.raises(NoIncidence, match=r".*Point\(42, 0\).*"):
            Transformation.from_points_and_conics([p1, p2, p3], [q1, Point(42, 0), q3], c1, c2)


class TestTransformationCollection:
    def test_translation(self) -> None:
        p = Point(0, 1)
        t = TransformationCollection([translation(2, 1)] * 2)
        result = t * p

        assert isinstance(result, PointCollection)
        assert result == PointCollection([Point(2, 2), Point(2, 2)])

    def test_inverse(self) -> None:
        E = TransformationCollection([np.eye(4)] * 10)
        M = TransformationCollection([rotation(np.pi, axis=Point(0, 1, 0))] * 10)
        assert M.inverse() * M == E

    def test_pow(self) -> None:
        t = TransformationCollection([translation(1, 2)] * 10)

        assert t**0 == TransformationCollection([np.eye(3)] * 10)
        assert t**1 == t
        assert t**2 == TransformationCollection([translation(2, 4)] * 10)
        assert t**3 == TransformationCollection([translation(3, 6)] * 10)
        assert t ** (-2) == TransformationCollection([translation(-2, -4)] * 10)
