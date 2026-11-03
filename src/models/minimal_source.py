from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int =  Field(ge=0)
