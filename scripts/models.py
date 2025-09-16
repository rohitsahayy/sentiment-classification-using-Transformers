from pydantic import BaseModel
from pydantic import HttpUrl,EmailStr
import time


class NLPDataInput(BaseModel):
    text : list[str]
    user_id :str

class NLPDataOutput(BaseModel):
    model_name : str
    text : list[str]
    labels : list[str]
    scores : list[float]
    prediction_time : int