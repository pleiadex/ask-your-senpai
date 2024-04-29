# CRAG implementation

## define embedding, flow


# TODO: just link to the open ai api


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

def get_openai_response(question: str):
    # Chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = (
        llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


# # Question
# rag_chain.invoke("What is Task Decomposition?")

