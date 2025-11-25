from aif360.algorithms.inprocessing import ExponentiatedGradientReduction

from .wrappers.exp_gradient_reduction_wrapper import ExpGradientReductionWrapper


def get_exponentiated_gradient_reduction_wrapper(estimator, inprocessor_configs, sensitive_attr_for_intervention):
    debiased_model = ExponentiatedGradientReduction(estimator=estimator,
                                                    constraints=inprocessor_configs['constraints'],
                                                    max_iter=inprocessor_configs['max_iter'],
                                                    run_linprog_step=inprocessor_configs['run_linprog_step'],
                                                    drop_prot_attr=inprocessor_configs['drop_prot_attr'],)
    models_config = {
        "ExponentiatedGradientReduction": ExpGradientReductionWrapper(inprocessor=debiased_model,
                                                                      sensitive_attr_for_intervention=sensitive_attr_for_intervention)
    }
    
    return models_config
