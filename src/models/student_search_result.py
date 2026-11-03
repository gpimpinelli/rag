from pydantic import BaseModel
from .minimal_search_result import MinimalSearchResults


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int