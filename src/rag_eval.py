import os
import json
from dotenv import load_dotenv
from database_manager import DatabaseManager
from rag_manager import RAGManager


load_dotenv()

# build vector database
chroma_path = f'./tmp/ap_textbooks/chroma.db'

# get the embedding function
embedding_function = DatabaseManager.get_embedding_function()

subjects = ['us_history', 'world_history', 'biology',  'macro_economics', 'psychology']

total_results = {}

for subject in subjects:
    qa_dataset = {
        'question': [],
        'contexts': [],
        'answer': [],
        'ground_truth': []
    }

    with open(f'./data/ap_exams/{subject}.json') as f:

        vanilla_output_file_path = f'./data/outputs/{subject}_output_vanilla.csv'
        output_directory = os.path.dirname(vanilla_output_file_path)
        os.makedirs(output_directory, exist_ok=True)

        with open(vanilla_output_file_path, 'w') as output_file:
                output_file.write("id, answer, ground_truth\n")

        #create an output text file
        advanced_output_file_path = f'./data/outputs/{subject}_output.csv'


        with open(advanced_output_file_path, 'w') as output_file:
                output_file.write("id, answer, ground_truth\n")

        data = json.load(f)
        for item in data['qa-dataset']:
            qa_dataset['question'].append(item['question'])
            qa_dataset['ground_truth'].append(item['ground_truth'])

            response1, sources1, contexts1 = RAGManager(
                                                chroma_path,
                                                embedding_function, 
                                                True,   # is_ap
                                                5,      # top_k
                                                True,   # compression
                                                True    # rerank
                                            ).run(item['question'])

            response2, sources2, contexts2 = RAGManager(
                                                chroma_path,
                                                embedding_function, 
                                                True,   # is_ap
                                                5,      # top_k
                                                False,   # compression
                                                False    # rerank
                                            ).get_answer_w_vanilla_rag(item['question'])

            response1 = response1.replace('\n', ' ')
            response2 = response2.replace('\n', ' ')

            with open(vanilla_output_file_path, 'a') as output_file:              
                # write response and ground truth to the output file
                output_file.write(f"{item['id']}, {response1}, {item['ground_truth']}\n")

            with open(advanced_output_file_path, 'a') as output_file:
                # write response and ground truth to the output file
                output_file.write(f"{item['id']}, {response2}, {item['ground_truth']}\n")