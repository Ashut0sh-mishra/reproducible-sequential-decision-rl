from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    filename: str

class ExperimentCreate(BaseModel):
    degradation_cost: float
    timesteps: int = 50000

class ExperimentStatus(BaseModel):
    task_id: str
    status: str
    result: dict | None
