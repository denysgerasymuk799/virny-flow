import numpy as np
import pandas as pd

from .abstract_error_injector import AbstractErrorInjector


class NullsInjector(AbstractErrorInjector):
    def __init__(self, seed: int):
        super().__init__(seed)
        np.random.seed(seed)

        self.columns_with_nulls = None
        self.null_percentage = None
        self.condition = None
        self.columns_in_query = None

    def _validate_input_params(self, df: pd.DataFrame, columns_with_nulls: list, null_percentage: float):
        if not 0.0 < null_percentage < 1.0:
            raise ValueError('The null_percentage parameter must be float from the (0.0-1.0) range.')

        # Check if all columns exist in the DataFrame
        if self.columns_in_query is not None and not self.columns_in_query.issubset(df.columns):
            raise ValueError("Query references columns that do not exist in the DataFrame.")

        for col in columns_with_nulls:
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Columns in columns_with_nulls must be the dataframe column names.")

    def _inject_nulls(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)

        existing_nulls_count = df_copy[self.columns_with_nulls].isna().any().sum()
        target_nulls_count = int(df_copy.shape[0] * self.null_percentage)

        if existing_nulls_count > target_nulls_count:
            raise ValueError(f"Existing nulls count in '{self.columns_with_nulls}' is greater than target nulls count. "
                             f"Increase nulls percentage for '{self.columns_with_nulls}' to be greater than existing nulls percentage.")

        nulls_sample_size = target_nulls_count - existing_nulls_count
        notna_idxs = df_copy[df_copy[self.columns_with_nulls].notna()].index

        random_row_idxs = np.random.choice(notna_idxs, size=nulls_sample_size, replace=False)
        random_columns = np.random.choice(self.columns_with_nulls, size=nulls_sample_size, replace=True)

        random_sample_df = pd.DataFrame({'column': random_columns, 'random_idx': random_row_idxs})
        for idx, col_name in enumerate(self.columns_with_nulls):
            col_random_row_idxs = random_sample_df[random_sample_df['column'] == col_name]['random_idx'].values
            if col_random_row_idxs.shape[0] == 0:
                continue

            df_copy.loc[col_random_row_idxs, col_name] = np.nan

        return df_copy

    def _inject_nulls_by_condition(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        # Apply the condition using query
        subset_df = df_copy.query(self.condition) if self.condition is not None else df_copy
        # Inject nulls
        subset_df_injected = self._inject_nulls(subset_df)
        df_copy.loc[subset_df_injected.index, :] = subset_df_injected

        return df_copy

    def fit(self, df: pd.DataFrame, columns_with_nulls: list, null_percentage: float, condition: str = None):
        if condition is not None:
            # Extracting the columns from the query string
            self.columns_in_query = {word for word in condition.split() if word in df.columns}
        self._validate_input_params(df=df, columns_with_nulls=columns_with_nulls, null_percentage=null_percentage)

        self.columns_with_nulls = columns_with_nulls
        self.null_percentage = null_percentage
        self.condition = condition

    def transform(self, df: pd.DataFrame):
        self._validate_input_params(df=df,
                                    columns_with_nulls=self.columns_with_nulls,
                                    null_percentage=self.null_percentage)

        return self._inject_nulls_by_condition(df)

    def fit_transform(self, df: pd.DataFrame, columns_with_nulls: list, null_percentage: float, condition: str = None):
        self.fit(df=df,
                 columns_with_nulls=columns_with_nulls,
                 null_percentage=null_percentage,
                 condition=condition)

        return self.transform(df)
