import os
import pathlib
from sklearn.model_selection import train_test_split


class S3Client:
    def __init__(self):
        pass

    def load_imputed_train_test_sets(self, data_loader, null_imputer_name: str, evaluation_scenario: str,
                                     experiment_seed: int):
        if self.test_set_fraction < 0.0 or self.test_set_fraction > 1.0:
            raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

        # Split the dataset
        y_train_val, y_test = train_test_split(data_loader.y_data,
                                               test_size=self.test_set_fraction,
                                               random_state=experiment_seed)

        # Read imputed train and test sets from save_sets_dir_path
        save_sets_dir_path = (pathlib.Path(__file__).parent.parent.parent
                              .joinpath('results')
                              .joinpath('imputed_datasets')
                              .joinpath(self.dataset_name)
                              .joinpath(null_imputer_name)
                              .joinpath(evaluation_scenario)
                              .joinpath(str(experiment_seed)))

        # Create a base flow dataset for Virny to compute metrics
        numerical_columns_wo_sensitive_attrs = [col for col in data_loader.numerical_columns if col not in self.dataset_sensitive_attrs]
        categorical_columns_wo_sensitive_attrs = [col for col in data_loader.categorical_columns if col not in self.dataset_sensitive_attrs]

        # Read X_train_val set
        train_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_train_val.csv'
        X_train_val_imputed_wo_sensitive_attrs = pd.read_csv(os.path.join(save_sets_dir_path, train_set_filename),
                                                             header=0, index_col=0)
        X_train_val_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs] = (
            X_train_val_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs].astype(str))

        # Subset y_train_val to align with X_train_val_imputed_wo_sensitive_attrs
        if null_imputer_name == ErrorRepairMethod.deletion.value:
            y_train_val = y_train_val.loc[X_train_val_imputed_wo_sensitive_attrs.index]

        # Read X_test sets
        X_tests_imputed_wo_sensitive_attrs_lst = list()
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        for test_injection_scenario in test_injection_scenarios_lst:
            test_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_test_{test_injection_scenario}.csv'
            X_test_imputed_wo_sensitive_attrs = pd.read_csv(os.path.join(save_sets_dir_path, test_set_filename),
                                                            header=0, index_col=0)
            X_test_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs] = (
                X_test_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs].astype(str))
            X_tests_imputed_wo_sensitive_attrs_lst.append(X_test_imputed_wo_sensitive_attrs)

        # Create base flow datasets for Virny to compute metrics
        main_base_flow_dataset, extra_base_flow_datasets = \
            create_virny_base_flow_datasets(data_loader=data_loader,
                                            dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                            X_train_val_wo_sensitive_attrs=X_train_val_imputed_wo_sensitive_attrs,
                                            X_tests_wo_sensitive_attrs_lst=X_tests_imputed_wo_sensitive_attrs_lst,
                                            y_train_val=y_train_val,
                                            y_test=y_test,
                                            numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                            categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return main_base_flow_dataset, extra_base_flow_datasets
