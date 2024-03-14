from pydantic import BaseModel
import numpy as np


class TaskRequest(BaseModel):
    task: str
    imgPath: str
