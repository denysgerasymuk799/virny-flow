"""Alpine Meadow pipeline (logical)."""

import pickle
import json

from google.protobuf.json_format import MessageToJson, Parse

from alpine_meadow.common.proto import pipeline_pb2


class Pipeline:
    """
    The class for a logical pipeline, consisting of many steps where a step is a concrete primitive
    """

    def __init__(self, **kwargs):
        # metadata
        self.id = None
        self.steps = []
        self.configuration = None
        self.tags = {}

        # runtime
        self.pipeline_arm = None
        self.metrics = None
        self.evaluated = False
        self.created_time = None

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def to_pipeline_desc(self, human_readable=False):
        """
        Return a human readable pipeline (with pretty format) if human_readable is True, otherwise return a pipeline
        for the backend execution engine
        """

        if human_readable:
            human_readable_steps = []
            for step in self.steps:
                human_readable_parameters = {key: f'{pickle.loads(value)}'
                                             for key, value in step.primitive.parameters.items()}
                human_readable_primitive = pipeline_pb2.Primitive(name=step.primitive.name,
                                                                  human_readable_parameters=human_readable_parameters)
                human_readable_steps.append(pipeline_pb2.Step(primitive=human_readable_primitive,
                                                              inputs=step.inputs, outputs=step.outputs))

            pipeline_desc = pipeline_pb2.PipelineDesc(steps=human_readable_steps)
        else:
            pipeline_desc = pipeline_pb2.PipelineDesc(steps=self.steps)

        return pipeline_desc

    @classmethod
    def from_pipeline_desc(cls, pipeline_desc, is_json=False, human_readable=False):
        """
        Return a new pipeline class from the pipeline description
        """

        if is_json:
            pipeline_desc = Parse(pipeline_desc, pipeline_pb2.PipelineDesc())

        def parse_value(string_value):
            string_value = string_value.strip()

            if string_value == 'True':
                return True
            if string_value == 'False':
                return False

            try:
                return int(string_value)
            except ValueError:
                pass

            try:
                return float(string_value)
            except ValueError:
                pass

            if string_value[0] == "'" and string_value[-1] == "'":
                return string_value[1:-1]

            try:
                if string_value[0] == '[' and string_value[-1] == ']':
                    return list(map(parse_value, string_value[1:-1].split(',')))
            except ValueError:
                pass

            return string_value

        if human_readable:
            steps = []
            for step in pipeline_desc.steps:
                parameters = {key: pickle.dumps(parse_value(value))
                              for key, value in step.primitive.human_readable_parameters.items()}
                primitive = pipeline_pb2.Primitive(name=step.primitive.name, parameters=parameters)
                steps.append(pipeline_pb2.Step(primitive=primitive, inputs=step.inputs))
            pipeline = Pipeline(steps=steps)
        else:
            pipeline = Pipeline(steps=pipeline_desc.steps)
        return pipeline

    def to_json(self):
        """
        Return the pipeline description in json format
        """

        return MessageToJson(self.to_pipeline_desc(human_readable=True))

    def dumps(self):
        dict_ = json.loads(self.to_json())
        dict_['id'] = self.id
        dict_['tags'] = self.tags
        dict_['created_time'] = self.created_time
        dict_['primitives'] = self.pipeline_arm.get_unique_primitives_strs()

        return dict_
