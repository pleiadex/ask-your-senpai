import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your senpai", page_icon="🤓")
    st.header("Ask your smart senpai 🤓")

    # Initialize the chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    # Display the chat history  

    # TODO: add submit-icon button for user input

    # TODO: add image to each message

    user_question = st.text_input("Ask a question about your textbooks:")
    if user_question:
        # TODO: invoke the chain
        pass

    with st.sidebar:
        st.subheader("Textbook Library")
        pdf_docs = st.file_uploader(
            "Upload textbooks on which you want to inquire", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # TODO: preprocess the pdfs

                # TODO: update the vector database
                pass

if __name__ == '__main__':
    main()