import yaml
from munch import DefaultMunch

from virny_flow.configs.constants import ErrorRepairMethod, FairnessIntervention, MLModels


def is_in_enum(val, enum_obj):
    enum_vals = [member.value for member in enum_obj]
    return val in enum_vals


def validate_config(exp_config_obj):
    """
    Validate parameter types and values in the exp_config_obj.
    """
    # ============================================================================================================
    # Required parameters
    # ============================================================================================================
    if not isinstance(exp_config_obj.exp_config_name, str):
        raise ValueError('exp_config_name must be string')

    if not isinstance(exp_config_obj.dataset, str):
        raise ValueError('dataset argument must be string')

    if not isinstance(exp_config_obj.sensitive_attrs_for_intervention, list):
        raise ValueError('sensitive_attrs_for_intervention must be a list')

    if not isinstance(exp_config_obj.random_state, int):
        raise ValueError('random_state must be integer')

    # Check list types
    if not isinstance(exp_config_obj.null_imputers, list):
        raise ValueError('null_imputers argument must be a list')

    if not isinstance(exp_config_obj.fairness_interventions, list):
        raise ValueError('fairness_interventions argument must be a list')

    if not isinstance(exp_config_obj.models, list):
        raise ValueError('models argument must be a list')

    for null_imputer_name in exp_config_obj.null_imputers:
        if not is_in_enum(val=null_imputer_name, enum_obj=ErrorRepairMethod):
            raise ValueError('null_imputers argument should include values from the ErrorRepairMethod enum in domain_logic/constants.py')

    for fairness_intervention in exp_config_obj.fairness_interventions:
        if not is_in_enum(val=fairness_intervention, enum_obj=FairnessIntervention):
            raise ValueError('fairness_interventions argument should include values from the FairnessIntervention enum in domain_logic/constants.py')

    for model_name in exp_config_obj.models:
        if not is_in_enum(val=model_name, enum_obj=MLModels):
            raise ValueError('models argument should include values from the MLModels enum in domain_logic/constants.py')

    return True


def create_exp_config_obj(config_yaml_path: str):
    """
    Return a config object created based on a config yaml file.

    Parameters
    ----------
    config_yaml_path
        Path to a config yaml file

    """
    with open(config_yaml_path) as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader)

    config_obj = DefaultMunch.fromDict(config_dct)
    validate_config(config_obj)

    return config_obj
