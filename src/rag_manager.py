from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate

class RAGManager:
    @staticmethod
    def get_answer(chroma_path:str, embedding_function, question: str):

        db = Chroma(
            persist_directory=chroma_path, 
            embedding_function=embedding_function
        )

        # TODO: Adaptive RAG

        results = db.similarity_search_with_score(question, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = hub.pull("rlm/rag-prompt") # FIXME: hard-coded since it is time-consuming
        prompt = prompt_template.format(context=context_text, question=question)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        chain = (
            llm |
            StrOutputParser()
        )

        # add refereces to response
        sources = [f'{doc.metadata.get("id", None)}:{_score}' for doc, _score in results]

        response = chain.invoke(prompt)

        return response, sources