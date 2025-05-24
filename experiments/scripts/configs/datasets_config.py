from virny.datasets import (DiabetesDataset2019, GermanCreditDataset, ACSEmploymentDataset, ACSIncomeDataset,
                            LawSchoolDataset, CardiovascularDiseaseDataset, ACSPublicCoverageDataset)


DATASET_CONFIG = {
    "diabetes": {
        "data_loader": DiabetesDataset2019,
        "data_loader_kwargs": {'with_nulls': False},
        "test_set_fraction": 0.3,
    },
    "german": {
        "data_loader": GermanCreditDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3,
    },
    "folk_emp": {
        "data_loader": ACSEmploymentDataset,
        "data_loader_kwargs": {"state": ['CA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 15_000, "subsample_seed": 42},
        "test_set_fraction": 0.2,
    },
    "folk_emp_big": {
        "data_loader": ACSEmploymentDataset,
        "data_loader_kwargs": {"state": ['CA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 200_000, "subsample_seed": 42},
        "test_set_fraction": 0.2,
    },
    "folk_inc": {
        "data_loader": ACSIncomeDataset,
        "data_loader_kwargs": {"state": ['GA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 15_000, "subsample_seed": 42},
        "test_set_fraction": 0.2,
    },
    "folk_pubcov": {
        "data_loader": ACSPublicCoverageDataset,
        "data_loader_kwargs": {"state": ['NY'], "year": 2018, "with_nulls": False,
                               "subsample_size": 50_000, "subsample_seed": 42},
        "test_set_fraction": 0.2,
    },
    "law_school": {
        "data_loader": LawSchoolDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2,
    },
    "heart": {
        "data_loader": CardiovascularDiseaseDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2,
    },
}
