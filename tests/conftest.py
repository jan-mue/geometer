import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(seed=0)
