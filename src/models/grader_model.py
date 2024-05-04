from langchain_core.pydantic_v1 import BaseModel, Field
from constants.answers import YES, NO

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description=f"Documents are relevant to the question, '{YES}' or '{NO}'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description=f"Answer is grounded in the facts, '{YES}' or '{NO}'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description=f"Answer addresses the question, '{YES}' or '{NO}'")
