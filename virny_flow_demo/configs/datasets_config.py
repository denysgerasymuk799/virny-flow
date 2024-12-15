from virny.datasets import DiabetesDataset2019, ACSEmploymentDataset
from virny_flow_demo.configs.data_loaders import GermanCreditDataset


DATASET_CONFIG = {
    "german": {
        "data_loader": GermanCreditDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3,
    },
    "diabetes": {
        "data_loader": DiabetesDataset2019,
        "data_loader_kwargs": {'with_nulls': False},
        "test_set_fraction": 0.3,
    },
    "folk_emp": {
        "data_loader": ACSEmploymentDataset,
        "data_loader_kwargs": {"state": ['CA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 15_000, "subsample_seed": 42},
        "test_set_fraction": 0.2,
    },
}
