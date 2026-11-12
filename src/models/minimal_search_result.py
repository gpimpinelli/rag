from pydantic import BaseModel
from .minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    "Associa a una domanda la lista dei chunk recuperati da BM25."
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]
