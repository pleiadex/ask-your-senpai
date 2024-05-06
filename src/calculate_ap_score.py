import csv

def compare_answers(answer, ground_truth):
    return answer == ground_truth

subjects = ['us_history']

scores = {}

for subject in subjects:
    rag_count = 0
    llm_count = 0

    csv_file_path = f'./data/outputs/{subject}_output.csv'

    # Open the CSV file
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Access the data in each row
            # Example: print the first column of each row
            answer = row[1]
            ground_truth = row[2]
            if compare_answers(answer, ground_truth):
                rag_count += 1


    with open(f'./data/outputs/{subject}_llm_output.csv', 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            answer = row[1]
            ground_truth = row[2]
            if compare_answers(answer, ground_truth):
                llm_count += 1
  
    scores[subject] = (rag_count, llm_count)

print(scores)