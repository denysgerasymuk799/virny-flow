import pathlib

from virny_flow_demo.configs.data_loaders import GermanCreditDataset


DATASET_CONFIG = {
    "german": {
        "data_loader": GermanCreditDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'german_config.yaml')
    },
}
