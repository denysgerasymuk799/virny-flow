import pytest

from virny.datasets import GermanCreditDataset


@pytest.fixture(scope="function")
def common_seed():
    return 42


# Fixture to load the dataset
@pytest.fixture(scope="function")
def german_data_loader():
    return GermanCreditDataset()
