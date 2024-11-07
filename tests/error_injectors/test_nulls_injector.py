import numpy as np

from virny_flow.core.error_injectors.nulls_injector import NullsInjector


def test_nulls_injector_mcar_strategy(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader
    mcar_injector = NullsInjector(seed=common_seed)
    injected_df = mcar_injector.fit_transform(df=data_loader.X_data,
                                              columns_with_nulls=["AGEP", "SCHL", "MAR"],
                                              null_percentage=0.3)

    np.testing.assert_almost_equal(injected_df.isnull().sum().sum() / injected_df.shape[0], 0.3, err_msg="MCAR strategy failed")


def test_nulls_injector_mar_strategy(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader
    mar_injector = NullsInjector(seed=common_seed)
    injected_df = mar_injector.fit_transform(df=data_loader.X_data,
                                             columns_with_nulls=["AGEP", "MAR"],
                                             null_percentage=0.3,
                                             condition='SEX == "2"')
    mask = injected_df["SEX"] == "2"

    np.testing.assert_almost_equal(injected_df[mask].isnull().sum().sum() / mask.sum(), 0.3, decimal=3, err_msg="MAR strategy failed")


def test_nulls_injector_mnar_strategy(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader
    mnar_injector = NullsInjector(seed=common_seed)
    injected_df = mnar_injector.fit_transform(df=data_loader.X_data,
                                              columns_with_nulls=["SEX"],
                                              null_percentage=0.3,
                                              condition='SEX == "2"')
    mask = data_loader.X_data["SEX"] == "2"

    np.testing.assert_almost_equal(injected_df[mask].isnull().sum().sum() / mask.sum(), 0.3, decimal=3, err_msg="MNAR strategy failed")


def test_nulls_injector_mnar_strategy_lst_conditional_val(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader
    mnar_injector = NullsInjector(seed=common_seed)
    injected_df = mnar_injector.fit_transform(df=data_loader.X_data,
                                              columns_with_nulls=["AGEP"],
                                              null_percentage=0.3,
                                              condition=f'AGEP <= 50')
    mask = data_loader.X_data["AGEP"] <= 50

    np.testing.assert_almost_equal(injected_df[mask].isnull().sum().sum() / mask.sum(), 0.3, decimal=3, err_msg="MNAR strategy failed")


def test_nulls_injector_mnar_strategy_lt_conditional_val(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader
    mnar_injector = NullsInjector(seed=common_seed)
    injected_df = mnar_injector.fit_transform(df=data_loader.X_data,
                                              columns_with_nulls=["WKHP"],
                                              null_percentage=0.3,
                                              condition='WKHP < 40.0')
    mask = data_loader.X_data["WKHP"] < 40.0

    np.testing.assert_almost_equal(injected_df[mask].isnull().sum().sum() / mask.sum(), 0.3, decimal=3, err_msg="MNAR strategy failed")


def test_nulls_injector_mnar_strategy_ge_conditional_val(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader
    mnar_injector = NullsInjector(seed=common_seed)
    injected_df = mnar_injector.fit_transform(df=data_loader.X_data,
                                              columns_with_nulls=["WKHP"],
                                              null_percentage=0.3,
                                              condition='WKHP >= 40.0')
    mask = data_loader.X_data["WKHP"] >= 40.0

    np.testing.assert_almost_equal(injected_df[mask].isnull().sum().sum() / mask.sum(), 0.3, decimal=3, err_msg="MNAR strategy failed")


def test_nulls_injector_same_seed_same_output(acs_income_dataloader, common_seed):
    data_loader = acs_income_dataloader

    mcar_injector1 = NullsInjector(seed=common_seed)
    injected_df1 = mcar_injector1.fit_transform(df=data_loader.X_data,
                                                columns_with_nulls=["AGEP", "SCHL", "MAR"],
                                                null_percentage=0.3)

    mcar_injector2 = NullsInjector(seed=common_seed)
    injected_df2 = mcar_injector2.fit_transform(df=data_loader.X_data,
                                                columns_with_nulls=["AGEP", "SCHL", "MAR"],
                                                null_percentage=0.3)

    assert injected_df1.equals(injected_df2), "Results from NullsInjector are not identical"
