from virny.datasets import (DiabetesDataset2019, ACSEmploymentDataset, ACSPublicCoverageDataset,
                            GermanCreditDataset, CardiovascularDiseaseDataset)

SEED = 42

DATASET_CONFIG = {
    "diabetes": {
        "data_loader": DiabetesDataset2019,
        "data_loader_kwargs": {'with_nulls': False},
        "test_set_fraction": 0.3,
        "virny_config": {
            "n_estimators": 50,
            "bootstrap_fraction": 0.8,
            "sensitive_attributes_dct": {'Gender': 'Female'},
        },
    },
    "german": {
        "data_loader": GermanCreditDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3,
        "virny_config": {
            "computation_mode": "no_bootstrap",
            "sensitive_attributes_dct": {'sex': 'female', 'age': [19, 20, 21, 22, 23, 24, 25], 'sex&age': None},
        },
    },
    "folk_emp": {
        "data_loader": ACSEmploymentDataset,
        "data_loader_kwargs": {"state": ['CA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 15_000, "subsample_seed": SEED},
        "test_set_fraction": 0.2,
        "virny_config": {
            "computation_mode": "no_bootstrap",
            "sensitive_attributes_dct": {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX&RAC1P': None},
        },
    },
    "folk_emp_big": {
        "data_loader": ACSEmploymentDataset,
        "data_loader_kwargs": {"state": ['CA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 200_000, "subsample_seed": SEED},
        "test_set_fraction": 0.2,
        "virny_config": {
            "computation_mode": "no_bootstrap",
            "sensitive_attributes_dct": {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX&RAC1P': None},
        },
    },
    "folk_pubcov": {
        "data_loader": ACSPublicCoverageDataset,
        "data_loader_kwargs": {"state": ['NY'], "year": 2018, "with_nulls": False,
                               "subsample_size": 50_000, "subsample_seed": SEED},
        "test_set_fraction": 0.2,
        "virny_config": {
            "computation_mode": "no_bootstrap",
            "sensitive_attributes_dct": {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX&RAC1P': None},
        },
    },
    "heart": {
        "data_loader": CardiovascularDiseaseDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2,
        "virny_config": {
            "computation_mode": "no_bootstrap",
            "sensitive_attributes_dct": {'gender': '1'},
        },
    },
}
