from constants.answers import YES, NO

LLM_FALLBACK_PREAMBLE = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

GENERATE_PREAMBLE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

LLM_FALLBACK_PREAMBLE_AP = """You are an assistant for advanced placement exam's question-answering tasks. Answer the question based upon your knowledge. Use either A, B, C or D to indicate the correct answer. If you don't know the answer, you must choose the most likely choice among A, B, C, D, or E! Even if you need more information, you must choose the most likely choice among A, B, C, D or E! there is not enough information, ignore the provided context, use your own knowledge."""

GENERATE_PREAMBLE_AP = """You are an assistant for advanced placement exam's question-answering tasks. Use the following pieces of retrieved context to answer the question. Use either A, B, C or D to indicate the correct answer. If you don't know the answer, or you think there is not enough information, ignore the provided context, use your own knowledge and choose the most likely choice among A, B, C, D or E! Even if you need more information, you must choose the most likely choice among A, B, C, D or E!"""

DOCUMENT_GRADER_PREAMBLE = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score '{YES}' or '{NO}' score to indicate whether the document is relevant to the question."""

topics = 'World history, the United States, and the history of the Americas'

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

PURE_LLM_PREAMBLE = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

PURE_LLM_PREAMBLE_AP = """You are an assistant for advanced placement exam's question-answering tasks. Answer the question based upon your knowledge. Use either A, B, C or D to indicate the correct answer. If you don't know the answer, you must choose the most likely choice among A, B, C, D or E! Even if you need more information, you must choose the most likely choice among A, B, C, D or E!. DO NOT PROVIDE ANY INFORMATION but the answer choice  A, B, C, D or E. JUST THE ANSWER CHOICE without parenthesis!"""

VANILLA_RAG_PREAMBLE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

VANILLA_RAG_PREAMBLE_AP = """You are an assistant for advanced placement exam's question-answering tasks. Answer the question based the following pieces of retrieved context. Use either A, B, C, D or E to indicate the correct answer!! If you don't know the answer, you must choose the most likely choice among A, B, C, D or E! Even if you need more information, you must choose the most likely choice among A, B, C, D or E!. DO NOT PROVIDE ANY INFORMATION but the answer choice A, B, C, D or E. JUST THE ANSWER CHOICE without parenthesis or new line!"""