"""Quality model."""


class QualityModel:
    """
    The cost model for predicting the quality of a pipeline arm.
    """

    def estimate_current_quality(self, pipeline_arm):
        """
        Return the estimated quality (mean/std) of the given pipeline arm for the current session.
        :param pipeline_arm:
        :return:
        """

        return pipeline_arm.quality_mean, pipeline_arm.quality_std

    def get_external_quality(self, pipeline_arm):
        """
        Return the history quality (mean/std) of the given pipeline arm from history (using meta-learning).
        :param pipeline_arm:
        :return:
        """

        return pipeline_arm.external_quality_mean, pipeline_arm.external_quality_std
