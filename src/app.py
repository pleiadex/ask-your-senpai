import time
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv

from roles import SENPAI, USER
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
        name, page, chunk, score = source.split(':')

        source_dict[name] = source_dict.get(name, [])
        source_dict[name].append(page)

    res = "I found the information in the following textbooks:  \n"

    for name, pages in source_dict.items():
        res += f"- `{name}`  \n  The most relevant pages:"

        page_counts = {}
        for page in pages:
            page_counts[page] = page_counts.get(page, 0) + 1

        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)

        for page, count in sorted_pages:
            res += f" {page}" # ({count} times)"
            if page != sorted_pages[-1][0]:
                res += ","

    return res


def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your senpai", page_icon="🤓")
    st.header("Ask your smart senpai 👀")

    # Initialize the chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chroma_path" not in st.session_state:
        uuid = str(uuid4())
        st.session_state.chroma_path = f'./tmp/{uuid}/chroma.db'

    chroma_path = st.session_state.chroma_path

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"].name, avatar=message["role"].avatar):
            st.markdown(message["content"])

    # Display the chat history  
    if prompt := st.chat_input("Ask a question about your textbooks"):
        
        st.session_state.chat_history.append({"role": USER, "content": prompt})
        with st.chat_message(USER.name, avatar=USER.avatar):
            st.markdown(prompt)

        # invoke the chain
        with st.chat_message(SENPAI.name, avatar=SENPAI.avatar):
            embedding_function = DatabaseManager.get_embedding_function()
            response, sources = RAGManager.get_answer(chroma_path, embedding_function, prompt)
            
            st.write_stream(response_generator(response))

            formatted_sources = source_formatter(sources)
            if sources is not None and len(sources) > 0:
                st.markdown(formatted_sources)

        st.session_state.chat_history.append({"role": SENPAI, "content": f"{response}\n\n{formatted_sources}"})

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
                DatabaseManager.update_vectorstore(chroma_path, chunks)

            success = st.success("Textbooks uploaded successfully")
            time.sleep(2)
            success.empty()
            

if __name__ == '__main__':
    main()