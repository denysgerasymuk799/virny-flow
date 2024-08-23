from dataclasses import dataclass
from sklearn.impute import SimpleImputer


@dataclass
class MixedImputer:
    num_imputer: SimpleImputer
    cat_imputer: SimpleImputer
