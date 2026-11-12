from .minimal_search_result import MinimalSearchResults


class MinimalAnswer(MinimalSearchResults):
    "Estende MinimalSearchResults aggiungendo la risposta generata dal LLM."
    answer: str
