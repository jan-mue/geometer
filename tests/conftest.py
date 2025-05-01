import numpy as np
import pytest
from numpy.random import Generator


@pytest.fixture(scope="session")
def rng() -> Generator:
    return np.random.default_rng(seed=0)
