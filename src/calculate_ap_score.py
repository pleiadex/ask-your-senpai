import csv

def compare_answers(answer, ground_truth):
    if answer == ground_truth:
        return True
    
    if f"({ground_truth.replace(' ', '')})" in answer:
        return True
    
    return False

subjects = ['us_history', 'world_history', 'biology',  'macro_economics', 'psychology']

scores = {}

for subject in subjects:
    rag_count = 0
    llm_count = 0
    vanilia_rag_count = 0


    csv_file_path = f'./data/outputs/{subject}_output.csv'

    # Open the CSV file
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Access the data in each row
            answer = row[1]
            ground_truth = row[-1]
            if compare_answers(answer, ground_truth):
                rag_count += 1


    with open(f'./data/outputs/{subject}_llm_output.csv', 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            answer = row[1]
            ground_truth = row[-1]
            if compare_answers(answer, ground_truth):
                llm_count += 1

    with open(f'./data/outputs/{subject}_output_vanilla.csv', 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            answer = row[1]
            ground_truth = row[-1]
            if compare_answers(answer, ground_truth):
                vanilia_rag_count += 1
  
    scores[subject] = (rag_count, llm_count, vanilia_rag_count)

print(scores)