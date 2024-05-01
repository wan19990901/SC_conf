from LLM_agent import *
import os
import pandas as pd
from tqdm import tqdm

DATA_DIR = '../data'

# Experiment Config

DF_NAME = 'MathQA'
DIFFICULTY = 'hard'

NUM_OF_SAMPLES = 100
NUM_OF_COT = 40
llm_config = {
    # change these three together
    'llm_type': 'openai',  # openai, ollama, anthropic
    'api_key_link': 'api_key_yw.txt',
    'model': "gpt-4",  # see llm_model.txt
    # change these two together
    'prompt_link': 'prompt_template.json',
    'parser_template': CoT,
    # change as needed
    'temperature': 0,
}


def save_csv(row):
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/{llm_config["model"]}')
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    file_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.csv')

    if not os.path.isfile(file_path):
        # The file does not exist, write with header
        row.to_frame().T.to_csv(file_path, mode='a', index=False, header=True)
    else:
        # The file exists, append without header
        row.to_frame().T.to_csv(file_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read()
    df = pd.read_csv(os.path.join(DATA_DIR, f'{DF_NAME}/{DF_NAME}_{DIFFICULTY}.csv'))
    df_subset = df[:NUM_OF_SAMPLES+2]

    # Data collection
    for row_idx in tqdm(range(101,len(df_subset)), colour='blue', desc='Sample Progress', position=0):
        row = df_subset.iloc[7]
        subject = row['Category']
        question = row['Question']

        arguments_dict = {
            'subject': subject,
            'question': question
        }

        for round in tqdm(range(NUM_OF_COT), colour='green', desc='Round', position=1):
            instruction_violation_count = []
            parse_error = True
            parse_error_attempt = 0
            while parse_error and parse_error_attempt < 3:
                cot_agent = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                                      temperature=llm_config['temperature'])
                cot_agent.set_prompt(llm_config['prompt_link'])
                cot_agent.set_parser(llm_config['parser_template'])
                response, attempts = cot_agent.involk(arguments_dict)
                try:
                    CoT = response['CoT']
                    Final_Answer = response['Final_answer']
                    parse_error = False
                except:
                    CoT = response
                    Final_Answer = 'error'
                    parse_error = True
                    parse_error_attempt += 1
                instruction_violation_count.append((attempts,parse_error_attempt))
            row[f'CoT_{round}'] = CoT
            row[f'Final Answer_{round}'] = Final_Answer
            row[f'Instruction Violation_{round}'] = instruction_violation_count

        save_csv(row)
