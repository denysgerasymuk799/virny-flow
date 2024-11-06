import copy
import pandas as pd

from .dataframe_utils import encode_cat, decode_cat, encode_cat_with_existing_encoder


def get_df_condition(df: pd.DataFrame, col: str, dis, include_dis: bool):
    if isinstance(dis, list):
        return df[col].isin(dis) if include_dis else ~df[col].isin(dis)
    else:
        return df[col] == dis if include_dis else df[col] != dis


def get_dis_group_condition(df, attrs, dis_values):
    # Construct a complex df condition for the dis group
    dis_condition = get_df_condition(df, attrs[0], dis_values[0], include_dis=True)
    for idx in range(1, len(attrs)):
        dis_condition &= get_df_condition(df, attrs[idx], dis_values[idx], include_dis=True)

    return dis_condition


def encode_dataset_for_missforest(df, cat_encoders: dict = None, dataset_name: str = None,
                                  categorical_columns_with_nulls: list = None):
    df_enc = copy.deepcopy(df)
    cat_columns = df.select_dtypes(include=['object']).columns

    if cat_encoders is None:
        cat_encoders = dict()
        for c in cat_columns:
            c_enc, encoder = encode_cat(df_enc[c])
            df_enc[c] = c_enc
            cat_encoders[c] = encoder
    else:
        for c in cat_columns:
            df_enc[c] = encode_cat_with_existing_encoder(df_enc[c], cat_encoders[c])

    # Get indices of categorical columns
    cat_indices = [df_enc.columns.get_loc(col) for col in cat_columns]

    return df_enc, cat_encoders, cat_indices


def decode_dataset_for_missforest(df_enc, cat_encoders, dataset_name: str = None):
    df_dec = copy.deepcopy(df_enc)

    for c in cat_encoders.keys():
        df_dec[c] = decode_cat(df_dec[c], cat_encoders[c])

    return df_dec


def nested_dict_from_flat(d: dict):
    result = {}
    for key, value in d.items():
        # Split the key by '__'
        parts = key.split("__")

        # Start at the top level of the result dictionary
        current_level = result
        for part in parts[:-1]:  # Go down to the last part
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Assign the value to the last key in parts
        current_level[parts[-1]] = value
    return result
