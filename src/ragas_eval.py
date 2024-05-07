import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from pdf_manager import PDFManager
from database_manager import DatabaseManager
from rag_manager import RAGManager

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

load_dotenv()

# build vector database
chroma_path = f'./tmp/papers/chroma.db'

paper_dir_path = './data/papers'

pdf_file_paths = [os.path.join(paper_dir_path, file) for file in os.listdir(paper_dir_path) if file.endswith('.pdf')]

pdf_docs = PDFManager.get_docs_from_text(pdf_file_paths)
chunks = DatabaseManager.split_documents(pdf_docs)
DatabaseManager.update_vectorstore(chroma_path, chunks)

# get the embedding function
embedding_function = DatabaseManager.get_embedding_function()

# df = pd.read_pickle('./data/papers/papers.pkl')
df = pd.read_csv('./data/papers/papers.csv')

vanilla_qa_dataset = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }

advanced_qa_dataset = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }

for index, row in df.iterrows():
    if type(row['ground_truth']) is not str:
        continue
    
    # vanilla rag
    response, sources, contexts = RAGManager(
                                        chroma_path,
                                        embedding_function, 
                                        False,   # is_ap
                                        5,      # top_k
                                        False,   # compression
                                        False    # rerank
                                    ).get_answer_w_vanilla_rag(row['question'])

    
    vanilla_qa_dataset['question'].append(row['question'])
    vanilla_qa_dataset['contexts'].append(contexts)
    vanilla_qa_dataset['answer'].append(response)
    vanilla_qa_dataset['ground_truth'].append(row['ground_truth'])

    # advanced rag
    response, sources, contexts = RAGManager(
                                        chroma_path,
                                        embedding_function, 
                                        False,   # is_ap
                                        5,      # top_k
                                        True,   # compression
                                        True    # rerank
                                    ).run(row['question'])
    
    advanced_qa_dataset['question'].append(row['question'])
    advanced_qa_dataset['contexts'].append(contexts)
    advanced_qa_dataset['answer'].append(response)
    advanced_qa_dataset['ground_truth'].append(row['ground_truth'])

print(vanilla_qa_dataset)
print(advanced_qa_dataset)

vanilla_dataset = Dataset.from_dict(vanilla_qa_dataset)
advanced_dataset = Dataset.from_dict(advanced_qa_dataset)

# run ragas evaluation
vanilla_result = evaluate(
    vanilla_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    raise_exceptions=False
)

advanced_result = evaluate(
    advanced_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    raise_exceptions=False
)

# save total_results to a text file
with open('./data/outputs/ragas_vanilla_results.txt', 'w') as f:
    f.write(str(vanilla_result))
    print(vanilla_result)

with open('./data/outputs/ragas_advanced_results.txt', 'w') as f:
    f.write(str(advanced_result))
    print(advanced_result)
