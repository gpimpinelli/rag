from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    "Rappresenta le coordinate esatte di un chunk all'interno del corpus."
    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
