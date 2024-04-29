import time
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv

from roles import SENPAI, USER
from rag_manager import get_openai_response
from pdf_manager import PDFManager
from database_manager import DatabaseManager


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

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

        # TODO: invoke the chain
        with st.chat_message(SENPAI.name, avatar=SENPAI.avatar):
            response = get_openai_response(prompt)
            st.write_stream(response_generator(response))
        st.session_state.chat_history.append({"role": SENPAI, "content": response})

    with st.sidebar:
        st.subheader("Textbook Library")
        pdf_docs = st.file_uploader(
            "Upload textbooks on which you want to inquire", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                docs = PDFManager.get_docs_from_text(pdf_docs)

                # TODO: embed images and tables
                images = PDFManager.extract_images(pdf_docs)
                tables = PDFManager.extract_tables(pdf_docs)

                # TODO: update the vector database
                chunks = DatabaseManager.split_documents(docs)
                DatabaseManager.update_vectorstore(chroma_path, chunks)


if __name__ == '__main__':
    main()