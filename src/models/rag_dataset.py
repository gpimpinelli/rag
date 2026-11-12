from pydantic import BaseModel
from .answered_question import AnsweredQuestion
from .unanswered_question import UnansweredQuestion


class RagDataset(BaseModel):
    "Contenitore per validare l'intero file JSON di un dataset di domande."
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
