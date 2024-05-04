from constants.answers import YES, NO

LLM_FALLBACK_PREAMBLE = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

GENERATE_PREAMBLE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

DOCUMENT_GRADER_PREAMBLE = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score '{YES}' or '{NO}' score to indicate whether the document is relevant to the question."""

topics = 'engineerings'

ROUTE_QUESTION_PREAMBLE_WITH_WEB_SEARCH = f"""You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to {topics}.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

ROUTE_QUESTION_PREAMBLE_WITHOUT_WEB_SEARCH = f"""You are an expert at routing a user question to a vectorstore.
The vectorstore contains documents related to {topics}.
Use the vectorstore for questions on these topics. Otherwise, use the fallback model."""

HALLUCINATION_GRADER_PREAMBLE = f"""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score '{YES}' or '{NO}'. '{YES}' means that the answer is grounded in / supported by the set of facts."""

ANSWER_GRADER_PREMABLE = f"""You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score '{YES}' or '{NO}'. '{YES}' means that the answer resolves the question."""
