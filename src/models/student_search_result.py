from pydantic import BaseModel
from .minimal_search_result import MinimalSearchResults


class StudentSearchResults(BaseModel):
    "Usati per creare i file JSON finali dei search_dataset e answer_dataset."
    search_results: list[MinimalSearchResults]
    k: int
