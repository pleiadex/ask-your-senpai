import os
import json
from dotenv import load_dotenv

from rag_manager import RAGManager

load_dotenv()


# initialize the dataset
subjects = ['biology'] #, 'macro_economics', 'psychology'] #['us_history', 'world_history', 

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
        for i, item in enumerate(data['qa-dataset']):
            response = RAGManager.get_answer_wo_rag(item['question'], True)

            print(f"{response}: {item['ground_truth']}")

            qa_dataset['question'].append(item['question'])
            qa_dataset['ground_truth'].append(item['ground_truth'])
            qa_dataset['answer'].append(response)

            # create an output text file
            output_file_path = f'./data/outputs/{subject}_llm_output.csv'
            output_directory = os.path.dirname(output_file_path)
            os.makedirs(output_directory, exist_ok=True)

            with open(output_file_path, 'a') as output_file:
              # write header if file is empty
              if os.stat(output_file_path).st_size == 0:
                output_file.write("id, answer, ground_truth\n")
              
              # write response and ground truth to the output file
              output_file.write(f"{item['id']}, {response}, {item['ground_truth']}\n")


    # save qa_dataset to a json file
    with open(f'./data/qa_dataset/{subject}_llm_qa_dataset.json', 'w') as f:
        json.dump(qa_dataset, f)


# TODO: visualize the results
