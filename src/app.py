import time
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv

from constants.roles import SENPAI, USER
from pdf_manager import PDFManager
from database_manager import DatabaseManager
from rag_manager import RAGManager


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def source_formatter(sources: list):
    source_dict = {}
    for source in sources:
        try:
            name, page, chunk = source.split(':')
    
        except AttributeError: # web-search
            return "I found this online."

        source_dict[name] = source_dict.get(name, [])
        source_dict[name].append(page)

    res = "I found the information in the following textbooks:  \n"

    for name, pages in source_dict.items():
        res += f"- `{name}`  \n  The most relevant pages:"

        page_counts = {}
        for page in pages:
            page_counts[page] = page_counts.get(page, 0) + 1

        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True) # reverse sort by count
        sorted_pages = sorted(sorted_pages, key=lambda x: x[0]) # sort by page number

        for page, count in sorted_pages:
            res += f" {page} ({count} times)"
            if page != sorted_pages[-1][0]:
                res += ","

    return res


def main():
    
    # Load environment variables in .env
    load_dotenv()

    # Initialize chat application
    st.set_page_config(page_title="Ask your senpai", page_icon="🤓")
    st.header("Ask your smart senpai 👀")

    # Initialize the chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize the ChromaDB file for this session
    if "chroma_path" not in st.session_state:
        uuid = str(uuid4())
        st.session_state.chroma_path = f'./tmp/{uuid}/chroma.db'

    # Enable a toggle for allowing web search
    is_web_search_enabled = st.toggle("Allow web search", True)
    
    enable_rag = st.toggle("Enable RAG", True)
    
    enable_compression = st.toggle("Enable compression", True)
    
    enable_rerank = st.toggle("Enable reranking", True)

    # Write all messages in the current chat history for this session to the UI
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"].name, avatar=message["role"].avatar):
            st.markdown(message["content"])
            
    with st.sidebar:
        num_docs_context = st.text_input("Number of vectors used as context")
        
        if num_docs_context:
            try:
                num_docs = int(num_docs_context)
                st.write(f"You entered: {num_docs}")
            except(ValueError):
                st.write(f"Invalid input")
        else:
            num_docs = 4
            
        

    # Get and process the next inputted prompt
    if prompt := st.chat_input("Ask a question about your textbooks"):
        
        # Add a text box to the UI for the USER prompt
        st.session_state.chat_history.append({"role": USER, "content": prompt})
        with st.chat_message(USER.name, avatar=USER.avatar):
            st.markdown(prompt)

        # Invoke the app (StateGraph)
        with st.chat_message(SENPAI.name, avatar=SENPAI.avatar):
            
            embedding_function = DatabaseManager.get_embedding_function()
            
            if enable_rag:
                response, sources, contexts = RAGManager(st.session_state.chroma_path, embedding_function, is_web_search_enabled, num_docs, enable_compression, enable_rerank).run(prompt)
            else:
                response, sources = RAGManager.get_answer()
            st.write_stream(response_generator(response))

            formatted_sources = ""
            if sources is not None and len(sources) > 0:
                formatted_sources = source_formatter(sources)
                st.markdown(formatted_sources)

            content = f"{response}\n\n{formatted_sources}" if formatted_sources else response

        st.session_state.chat_history.append({"role": SENPAI, "content": content})

    with st.sidebar:
        st.subheader("Textbook Library")
        pdf_docs = st.file_uploader(
            "Upload textbooks on which you want to inquire", accept_multiple_files=True)
        
        if st.button("Upload") and pdf_docs is not None and len(pdf_docs) > 0:
            with st.spinner("Processing"):
                docs = PDFManager.get_docs_from_text(pdf_docs)

                # TODO: embed images and tables
                # images = PDFManager.extract_images(pdf_docs)
                # tables = PDFManager.extract_tables(pdf_docs)

                # update the vector database
                chunks = DatabaseManager.split_documents(docs)
                DatabaseManager.update_vectorstore(st.session_state.chroma_path, chunks)

            success = st.success("Textbooks uploaded successfully")
            time.sleep(2)
            success.empty()

if __name__ == '__main__':
    main()