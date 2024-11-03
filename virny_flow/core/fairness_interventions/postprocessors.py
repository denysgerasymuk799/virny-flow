from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification


def get_eq_odds_postprocessor(privileged_groups, unprivileged_groups, seed=42):
    return EqOddsPostprocessing(privileged_groups=privileged_groups,
                                unprivileged_groups=unprivileged_groups,
                                seed=seed)
    
def get_reject_option_classification_postprocessor(privileged_groups, unprivileged_groups, postprocessor_configs):
    return RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups,
                                      **postprocessor_configs)
