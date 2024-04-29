from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class DatabaseManager:

    @staticmethod
    def split_documents(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
        )

        chunks = text_splitter.split_documents(documents)

        last_page_id = 0
        current_chunk_counter = 0
        for chunk in chunks:
            source = chunk.metadata.get('source', '')
            page_id = chunk.metadata.get('page_number', '')

            if page_id != last_page_id:
                current_chunk_counter = 0
                last_page_id = page_id

            current_page_id = f'{source}:{page_id}:{current_chunk_counter}'
            chunk.metadata['id'] = current_page_id
            
            current_chunk_counter += 1

        return chunks

    @staticmethod
    def update_vectorstore(chroma_path: str, chunks: list[Document]):
        
        db = Chroma(
                persist_directory=chroma_path,
                embedding_function=OpenAIEmbeddings()
            )
        
        existing_items = db.get(include=[])
        existing_ids = set(existing_items['ids'])

        print(f'The number of existing documents in DB: {len(existing_ids)}')

        new_chunks = []

        for chunk in chunks:
            if chunk.metadata['id'] not in existing_ids:
                new_chunks.append(chunk)

        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()




