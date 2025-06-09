from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.random import Generator


@pytest.fixture(scope="session")
def rng() -> Generator:
    return np.random.default_rng(seed=0)
