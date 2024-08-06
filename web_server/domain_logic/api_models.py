from pydantic import BaseModel


class GetWorkerTaskRequest(BaseModel):
    exp_config_name: str


class CompleteWorkerTaskRequest(BaseModel):
    task_id: str
    task_name: str
