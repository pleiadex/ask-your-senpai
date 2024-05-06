import os
import time
import json
from uuid import uuid4
from dotenv import load_dotenv

from datasets import Dataset

from ragas import evaluate

from pdf_manager import PDFManager
from database_manager import DatabaseManager
from rag_manager import RAGManager

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

load_dotenv()


# build vector database
chroma_path = f'./tmp/ap_textbooks/chroma.db'

# get the embedding function
embedding_function = DatabaseManager.get_embedding_function()


# initialize the dataset
# subjects = ['us_history', 'world_history', 'biology', 'macro_economics', 'psychology']
subjects = ['biology']
#, 'biology', 'macro_economics', 'psychology']

total_results = {}

for subject in subjects:
    qa_dataset = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }

    with open(f'./data/ap_exams/{subject}.json') as f:
        data = json.load(f)
        for item in data['qa-dataset'][-4:]:
            qa_dataset['question'].append(item['question'])
            qa_dataset['ground_truth'].append(item['ground_truth'])

            response, sources, contexts = RAGManager(
                                                chroma_path,
                                                embedding_function, 
                                                True,   # is_ap
                                                5,      # top_k
                                                False,   # compression
                                                False    # rerank
                                            ).run(item['question'])

            # response, sources, contexts = RAGManager(
            #                                     chroma_path,
            #                                     embedding_function, 
            #                                     True,   # is_ap
            #                                     5,      # top_k
            #                                     False,   # compression
            #                                     False    # rerank
            #                                 ).get_answer_w_vanilla_rag(item['question'])

            print(item['ground_truth'])

            response = response.replace('\n', ' ')

            qa_dataset['answer'].append(response)
            qa_dataset['contexts'].append(contexts)

            # create an output text file
            output_file_path = f'./data/outputs/{subject}_output.csv'
            output_directory = os.path.dirname(output_file_path)
            os.makedirs(output_directory, exist_ok=True)

            with open(output_file_path, 'a') as output_file:
                # write header if file is empty
                if os.stat(output_file_path).st_size == 0:
                    output_file.write("id, answer, ground_truth\n")
              
                # write response and ground truth to the output file
                output_file.write(f"{item['id']}, {response}, {item['ground_truth']}\n")


    # save qa_dataset to a json file
    with open(f'./data/qa_dataset/{subject}_qa_dataset_vanilla.json', 'w') as f:
        json.dump(qa_dataset, f)

    # # run ragas evaluation
    # result = evaluate(
    #     Dataset.from_dict(qa_dataset),
    #     metrics=[
    #         context_precision,
    #         faithfulness,
    #         answer_relevancy,
    #         context_recall,
    #     ],
    #     raise_exceptions=False
    # )

    # total_results[subject] = result

# # save total_results to a text file
# with open('./data/outputs/ragas_results.txt', 'w') as f:
#     f.write(str(total_results))
