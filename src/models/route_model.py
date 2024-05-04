from langchain_core.pydantic_v1 import BaseModel, Field

class WebSearch(BaseModel):
    """
    The internet. Use web_search for questions that are related to anything else than given topics.
    """
    query: str = Field(description="The query to use when searching the internet.")

class Vectorstore(BaseModel):
    """
    Use the vectorstore for questions on given topics.
    """
    query: str = Field(description="The query to use when searching the vectorstore.")