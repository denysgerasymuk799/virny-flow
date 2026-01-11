"""Implementation for primitive helper."""

from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.utils import ignore_warnings
from alpine_meadow.primitives.base import SKLearnPrimitive, XGBoostPrimitive


LOGICAL_TO_PHYSICAL_TABLE = {
    # data
    base.Primitive.ExtractColumnsByNames: 'alpine_meadow.primitives.feature_processing.ExtractColumnsByNames',
    base.Primitive.HorizontalConcat: 'alpine_meadow.primitives.feature_processing.HorizontalConcat',
    base.Primitive.FeatureEngineering: 'alpine_meadow.primitives.feature_processing.FeatureEngineering',

    # feature
    base.Primitive.Imputer: 'sklearn.impute.SimpleImputer',
    base.Primitive.MinMaxScaler: 'sklearn.preprocessing.MinMaxScaler',
    base.Primitive.StandardScaler: 'sklearn.preprocessing.StandardScaler',
    base.Primitive.RobustScaler: 'sklearn.preprocessing.RobustScaler',
    base.Primitive.LabelEncoder: 'alpine_meadow.primitives.feature_processing.UnseenLabelEncoder',
    # base.Primitive.OneHotEncoder: 'sklearn.preprocessing.OneHotEncoder',
    base.Primitive.OneHotEncoder: 'alpine_meadow.primitives.feature_processing.OneHotEncoder',
    base.Primitive.PCA: 'sklearn.decomposition.PCA',
    base.Primitive.KernelPCA: 'sklearn.decomposition.KernelPCA',
    base.Primitive.TruncatedSVD: 'sklearn.decomposition.TruncatedSVD',
    base.Primitive.FastICA: 'sklearn.decomposition.FastICA',
    base.Primitive.PolynomialFeatures: 'sklearn.preprocessing.PolynomialFeatures',
    base.Primitive.SelectPercentile: 'sklearn.feature_selection.SelectPercentile',
    base.Primitive.GenericUnivariateSelect: 'sklearn.feature_selection.GenericUnivariateSelect',
    # base.Primitive.SelectKBest: 'sklearn.feature_selection.SelectKBest', # missing
    base.Primitive.VarianceThreshold: 'sklearn.feature_selection.VarianceThreshold',
    base.Primitive.FeatureAgglomeration: 'sklearn.cluster.FeatureAgglomeration',
    base.Primitive.RBFSampler: 'sklearn.kernel_approximation.RBFSampler',
    base.Primitive.Normalizer: 'sklearn.preprocessing.Normalizer',
    base.Primitive.TimestampConverter: 'alpine_meadow.primitives.feature_processing.TimestampConverter',

    # classification
    base.Primitive.DecisionTreeClassifier: 'sklearn.tree.DecisionTreeClassifier',
    base.Primitive.LogisticRegression: 'sklearn.linear_model.LogisticRegression',
    base.Primitive.RandomForestClassifier: 'sklearn.ensemble.RandomForestClassifier',
    base.Primitive.XGradientBoostingClassifier: 'xgboost.XGBClassifier',
    base.Primitive.LGBMClassifier: 'lightgbm.LGBMClassifier',
    # base.Primitive.SVC: 'sklearn.svm.SVC',
    # base.Primitive.LinearSVC: 'sklearn.svm.LinearSVC',
    # base.Primitive.SGDClassifier: 'sklearn.linear_model.SGDClassifier',
    # base.Primitive.GaussianNB: 'sklearn.naive_bayes.GaussianNB',
    # base.Primitive.AdaBoostClassifier: 'sklearn.ensemble.AdaBoostClassifier',  # missing
    # base.Primitive.KNeighborsClassifier: 'sklearn.neighbors.KNeighborsClassifier',
    # base.Primitive.BaggingClassifier: 'sklearn.ensemble.BaggingClassifier',
    # base.Primitive.ExtraTreesClassifier: 'sklearn.ensemble.ExtraTreesClassifier',
    # base.Primitive.GradientBoostingClassifier: 'sklearn.ensemble.GradientBoostingClassifier',
    # base.Primitive.LinearDiscriminantAnalysis: 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
    # base.Primitive.QuadraticDiscriminantAnalysis: 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis',
    # base.Primitive.BasicClassifier: 'sklearn.dummy.DummyClassifier',
    # base.Primitive.CatBoostClassifier: 'catboost.CatBoostClassifier',

    # regression
    base.Primitive.SVR: 'sklearn.svm.SVR',
    base.Primitive.LinearSVR: 'sklearn.svm.LinearSVR',
    base.Primitive.LinearRegression: 'sklearn.linear_model.LinearRegression',  # missing
    base.Primitive.Ridge: 'sklearn.linear_model.Ridge',
    base.Primitive.SGDRegressor: 'sklearn.linear_model.SGDRegressor',
    base.Primitive.RandomForestRegressor: 'sklearn.ensemble.RandomForestRegressor',
    base.Primitive.GaussianProcessRegressor: 'sklearn.gaussian_process.GaussianProcessRegressor',
    base.Primitive.AdaBoostRegressor: 'sklearn.ensemble.AdaBoostRegressor',  # missing
    base.Primitive.KNeighborsRegressor: 'sklearn.neighbors.KNeighborsRegressor',
    base.Primitive.BaggingRegressor: 'sklearn.ensemble.BaggingRegressor',  # missing
    base.Primitive.ExtraTreesRegressor: 'sklearn.ensemble.ExtraTreesRegressor',
    base.Primitive.GradientBoostingRegressor: 'sklearn.ensemble.GradientBoostingRegressor',
    base.Primitive.XGradientBoostingRegressor: 'xgboost.XGBRegressor',
    base.Primitive.ARDRegression: 'sklearn.linear_model.ARDRegression',
    base.Primitive.DecisionTreeRegressor: 'sklearn.tree.DecisionTreeRegressor',
    base.Primitive.LGBMRegressor: 'lightgbm.LGBMRegressor',
    base.Primitive.RuleFit: 'rulefit.RuleFit',
    base.Primitive.BasicRegressor: 'sklearn.dummy.DummyRegressor',
    base.Primitive.CatBoostRegressor: 'catboost.CatBoostRegressor',

    # optimization
    base.Primitive.ThresholdingPrimitive: 'alpine_meadow.primitives.prediction.ThresholdingPrimitive',
}


def unpickle_parameters(pickled_parameters):
    import pickle

    parameters = {}
    for parameter_name, pickled_value in pickled_parameters.items():
        parameters[parameter_name] = pickle.loads(pickled_value)
    return parameters


@ignore_warnings
def get_primitive_from_step(step):
    """
    Create the primitive from the given step description.
    :param step:
    :return:
    """

    import importlib

    # get primitive name and parameters
    primitive_name = step.primitive.name
    primitive_parameters = unpickle_parameters(step.primitive.parameters)

    # build primitive
    primitive = None
    if primitive_name in LOGICAL_TO_PHYSICAL_TABLE:
        # get module path and klass name
        full_path = LOGICAL_TO_PHYSICAL_TABLE[primitive_name]
        module_path = '.'.join(full_path.split('.')[:-1])
        klass_name = full_path.split('.')[-1]

        # get class
        module = importlib.import_module(module_path)
        klass = getattr(module, klass_name)

        # get primitive
        if full_path.startswith('alpine_meadow'):
            parameters = primitive_parameters
            primitive = klass(**parameters)

        elif full_path.startswith('xgboost') or full_path.startswith('rulefit') or full_path.startswith('lightgbm'):
            parameters = primitive_parameters
            primitive = XGBoostPrimitive(klass(**parameters))

        else:
            parameters = primitive_parameters
            primitive = SKLearnPrimitive(klass(**parameters))

    if primitive is None:
        raise NotImplementedError(f'Unknown primitive: {primitive_name}')

    return primitive


def get_primitive_name_from_step(step):
    return str(base.Primitive.Name.Name(step.primitive.name))


def get_method_arguments(method):
    import inspect

    return set(inspect.signature(method).parameters.keys())


def get_all_primitives():
    return LOGICAL_TO_PHYSICAL_TABLE.keys()
