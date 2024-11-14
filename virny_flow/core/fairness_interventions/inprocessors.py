import tensorflow.compat.v1 as tf

from sklearn.linear_model import LogisticRegression
from aif360.algorithms.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction

from .wrappers.adversarial_debiasing_wrapper import AdversarialDebiasingWrapper
from .wrappers.exp_gradient_reduction_wrapper import ExpGradientReductionWrapper


def get_adversarial_debiasing_wrapper_config(privileged_groups, unprivileged_groups, 
                                             inprocessor_configs, sensitive_attr_for_intervention):
    session = tf.Session()
    tf.disable_eager_execution()
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                          unprivileged_groups=unprivileged_groups,
                                          scope_name=inprocessor_configs['scope_name'],
                                          debias=inprocessor_configs['debias'],
                                          adversary_loss_weight=inprocessor_configs['adversary_loss_weight'],
                                          num_epochs=inprocessor_configs['num_epochs'],
                                          batch_size=inprocessor_configs['batch_size'],
                                          classifier_num_hidden_units=inprocessor_configs['classifier_num_hidden_units'],
                                          sess=session)
    models_config = {
        "AdversarialDebiasing": AdversarialDebiasingWrapper(inprocessor=debiased_model,
                                                            sensitive_attr_for_intervention=sensitive_attr_for_intervention
                                                            )
    }
    
    return models_config


def get_exponentiated_gradient_reduction_wrapper(inprocessor_configs, sensitive_attr_for_intervention):
    estimator = LogisticRegression(**inprocessor_configs['estimator_params'])
    debiased_model = ExponentiatedGradientReduction(estimator=estimator,
                                                    constraints=inprocessor_configs['constraints'],
                                                    eps=inprocessor_configs['eps'],
                                                    max_iter=inprocessor_configs['max_iter'],
                                                    nu=inprocessor_configs['nu'],
                                                    eta0=inprocessor_configs['eta0'],
                                                    run_linprog_step=inprocessor_configs['run_linprog_step'],
                                                    drop_prot_attr=inprocessor_configs['drop_prot_attr'],)
    models_config = {
        "ExponentiatedGradientReduction": ExpGradientReductionWrapper(inprocessor=debiased_model,
                                                        sensitive_attr_for_intervention=sensitive_attr_for_intervention)
    }
    
    return models_config
