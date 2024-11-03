from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_simple_preprocessor(base_flow_dataset):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), base_flow_dataset.categorical_columns),
        ('num', StandardScaler(), base_flow_dataset.numerical_columns),
    ])


def preprocess_base_flow_dataset(base_flow_dataset):
    column_transformer = get_simple_preprocessor(base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    base_flow_dataset.X_train_val = column_transformer.fit_transform(base_flow_dataset.X_train_val)
    base_flow_dataset.X_test = column_transformer.transform(base_flow_dataset.X_test)

    return base_flow_dataset, column_transformer


def preprocess_mult_base_flow_datasets(main_base_flow_dataset, extra_base_flow_datasets):
    column_transformer = get_simple_preprocessor(main_base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df

    # Preprocess main_base_flow_dataset
    main_base_flow_dataset.X_train_val = column_transformer.fit_transform(main_base_flow_dataset.X_train_val)
    main_base_flow_dataset.X_test = column_transformer.transform(main_base_flow_dataset.X_test)

    # Preprocess extra_base_flow_datasets
    extra_test_sets = []
    for i in range(len(extra_base_flow_datasets)):
        extra_base_flow_datasets[i].X_test = column_transformer.transform(extra_base_flow_datasets[i].X_test)
        extra_test_sets.append((extra_base_flow_datasets[i].X_test,
                                extra_base_flow_datasets[i].y_test,
                                extra_base_flow_datasets[i].init_sensitive_attrs_df))

    return main_base_flow_dataset, extra_test_sets
