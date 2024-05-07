import os
import json
from dotenv import load_dotenv

from pdf_manager import PDFManager
from database_manager import DatabaseManager

load_dotenv()

# build vector database
chroma_path = f'./tmp/ap_textbooks/chroma.db'

textbook_dir_path = './data/ap_textbooks'

pdf_file_paths = [os.path.join(textbook_dir_path, file) for file in os.listdir(textbook_dir_path)]

pdf_docs = PDFManager.get_docs_from_text(pdf_file_paths)
chunks = DatabaseManager.split_documents(pdf_docs)
DatabaseManager.update_vectorstore(chroma_path, chunks)


