from pydantic import BaseModel
from typing import Optional

class SearchQuery(BaseModel):
    question: str 
    hnsw : Optional[bool] = False