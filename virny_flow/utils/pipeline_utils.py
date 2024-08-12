import copy
import pandas as pd

from virny_flow.utils.dataframe_utils import encode_cat, decode_cat, encode_cat_with_existing_encoder


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
