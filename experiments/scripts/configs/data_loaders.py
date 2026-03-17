from virny.datasets import CardiovascularDiseaseDataset
from virny.datasets.base import BaseDataLoader


class CVDAgeCohortDataset(BaseDataLoader):
    """
    Data loader that filters the Cardiovascular Disease dataset by age range
    to create age-based cohorts for distribution shift experiments.

    Wraps CardiovascularDiseaseDataset and applies an age filter specified
    by min_age and/or max_age (inclusive bounds).

    Parameters
    ----------
    min_age
        Minimum age (inclusive). If None, no lower bound is applied.
    max_age
        Maximum age (inclusive). If None, no upper bound is applied.
    """

    def __init__(self, min_age: int = None, max_age: int = None):
        base_dataset = CardiovascularDiseaseDataset()
        df = base_dataset.full_df.copy()

        old_num_rows = len(df)
        old_base_rate = df[base_dataset.target].mean()
        gender_counts_old = df['gender'].value_counts()
        old_gender_ratio = gender_counts_old.get('1', 0) / gender_counts_old.get('2', 1)

        if min_age is not None:
            df = df[df['age'] >= min_age]
        if max_age is not None:
            df = df[df['age'] <= max_age]

        df = df.reset_index(drop=True)

        new_num_rows = len(df)
        new_base_rate = df[base_dataset.target].mean()
        gender_counts_new = df['gender'].value_counts()
        new_gender_ratio = gender_counts_new.get('1', 0) / gender_counts_new.get('2', 1)

        age_label = self._format_age_label(min_age, max_age)
        print(f"[CVDAgeCohortDataset {age_label}] Number of rows: {old_num_rows} -> {new_num_rows}")
        print(f"[CVDAgeCohortDataset {age_label}] Base rate: {old_base_rate:.4f} -> {new_base_rate:.4f}")
        print(f"[CVDAgeCohortDataset {age_label}] Gender ratio ('1' to '2'): {old_gender_ratio:.4f} -> {new_gender_ratio:.4f}")

        super().__init__(
            full_df=df,
            target=base_dataset.target,
            numerical_columns=base_dataset.numerical_columns,
            categorical_columns=base_dataset.categorical_columns,
        )

    @staticmethod
    def _format_age_label(min_age, max_age):
        if min_age is not None and max_age is not None:
            return f"age {min_age}-{max_age}"
        elif min_age is not None:
            return f"age >= {min_age}"
        elif max_age is not None:
            return f"age <= {max_age}"
        return "all ages"
