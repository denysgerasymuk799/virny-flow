import copy
import numpy as np

import tensorflow.compat.v1 as tf

from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction

from .wrappers.adversarial_debiasing_wrapper import AdversarialDebiasingWrapper
from .wrappers.exp_gradient_reduction_wrapper import ExpGradientReductionWrapper

def get_adversarial_debiasing_wrapper_config(privileged_groups, unprivileged_groups, inprocessor_configs):
    session = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                          unprivileged_groups=unprivileged_groups,
                                          scope_name=inprocessor_configs['intervention_option'],
                                          debias=inprocessor_configs['debias'],
                                          num_epochs=inprocessor_configs['num_epochs'],
                                          sess=session)
    models_config = {
        "AdversarialDebiasing": AdversarialDebiasingWrapper(inprocessor=debiased_model,
                                                            sensitive_attr_for_intervention=sensitive_attr_for_intervention
                                                            )
    }
    
    return models_config

def get_exponentiated_gradient_reduction_wrapper(inprocessor_configs):
    estimator = LogisticRegression(**inprocessor_configs['estimator_params'])
    debiased_model = ExponentiatedGradientReduction(estimator=estimator,
                                                    constraints=inprocessor_configs['constraints'],
                                                    drop_prot_attr=inprocessor_configs['drop_prot_attr'],)
    models_config = {
        "ExponentiatedGradientReduction": ExpGradientReductionWrapper(inprocessor=debiased_model,
                                                        sensitive_attr_for_intervention=sensitive_attr_for_intervention)
    }
    
    return models_config