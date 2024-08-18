import pandas as pd

from virny_flow.configs.constants import ErrorType
from virny_flow.utils.custom_logger import get_logger
from virny_flow.error_injectors.nulls_injector import NullsInjector


class EvaluationScenarioBuilder:
    def __init__(self, exp_config_name: str, dataset_name: str, seed: int):
        self.exp_config_name = exp_config_name
        self.dataset_name = dataset_name
        self.seed = seed

        self._logger = get_logger(logger_name='evaluation_scenario_builder')

    def _validate_evaluation_scenario(self, evaluation_scenario: list):
        required_params = {'columns_to_inject', 'error_type', 'condition', 'error_rate'}
        for injection_scenario in evaluation_scenario:
            if not required_params.issubset(set(injection_scenario.keys())):
                raise ValueError(f'Not all parameters are defined in the input evaluation scenario. '
                                 f'The required parameters are {required_params}.')
            if not isinstance(injection_scenario['columns_to_inject'], str):
                raise ValueError('The columns_to_inject parameter should be string.')
            if not isinstance(injection_scenario['condition'], str):
                raise ValueError('The condition parameter must be string.')
            if not (isinstance(injection_scenario['error_rate'], float) and 0.0 < injection_scenario['error_rate'] < 1.0):
                raise ValueError('The error_rate parameter must be float from the (0.0-1.0) range.')
            if not (isinstance(injection_scenario['error_type'], str) and ErrorType.has_value(injection_scenario['error_type'])):
                raise ValueError('The error_type parameter must be a value from the ErrorType enum.')

    def implement_evaluation_scenario(self, df: pd.DataFrame, evaluation_scenario: list):
        # Validate user input
        self._validate_evaluation_scenario(evaluation_scenario)

        # Parse the input evaluation scenario
        parsed_evaluation_scenario = dict()
        for injection_scenario in evaluation_scenario:
            # Parse user inputs
            parsed_injection_scenario = {
                'columns_to_inject': injection_scenario['columns_to_inject'].split(','),
                'condition': injection_scenario['condition'],
                'error_rate': injection_scenario['error_rate'],
            }
            parsed_evaluation_scenario.setdefault(injection_scenario['error_type'], []).append(parsed_injection_scenario)

        # Inject nulls based on the evaluation scenario
        if ErrorType.missing_value.value in parsed_evaluation_scenario.keys():
            nulls_injector = NullsInjector(seed=self.seed)
            for injection_scenario in parsed_evaluation_scenario[ErrorType.missing_value.value]:
                df = nulls_injector.fit_transform(df=df,
                                                  columns_with_nulls=injection_scenario['columns_to_inject'],
                                                  null_percentage=injection_scenario['error_rate'],
                                                  condition=injection_scenario['condition'])

        return df
