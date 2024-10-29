import argparse
from LLM_agent import *
import os
import pandas as pd
from tqdm import tqdm

DATA_DIR = '../data/question_data/preprocessed/'

# Experiment Config
DF_NAME = 'GSM8K'
NUM_OF_SAMPLES = 100
NUM_OF_COT = 40

def get_llm_config(args):
    return {
        'llm_type': args.llm_type,
        'api_key_link': args.api_key_file,
        'model': args.model,
        'prompt_link': args.prompt_file,
        'parser_template': CoT,
        'temperature': args.temperature,
    }

def save_csv(row, llm_config):
    storage_dir =  f'../data/Evaluation_CoTs/{llm_config["model"]}'
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    file_path = os.path.join(storage_dir, f'{DF_NAME}.csv')

    # Add prompt file name to the row
    row['Prompt_File'] = os.path.basename(llm_config['prompt_link'])

    if not os.path.isfile(file_path):
        # The file does not exist, write with header
        row.to_frame().T.to_csv(file_path, mode='a', index=False, header=True)
    else:
        # The file exists, append without header
        row.to_frame().T.to_csv(file_path, mode='a', index=False, header=False)

def process_difficulty(llm_config, start_index=0):
    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read()
    df = pd.read_csv(os.path.join(DATA_DIR, f'{DF_NAME}.csv'))
    df_subset = df[:NUM_OF_SAMPLES]

    # Data collection
    for row_idx in tqdm(range(start_index, len(df_subset)), colour='blue', desc='Progress', position=0):
        row = df_subset.iloc[row_idx]
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
                if llm_config['llm_type'] != 'ollama':
                    cot_agent.set_parser(llm_config['parser_template'])
                response, attempts = cot_agent.involk(arguments_dict)
                if llm_config['llm_type'] == 'ollama':
                    response = response.content
                try:
                    CoT = response['CoT']
                    Final_Answer = response['Final_answer']
                    parse_error = False
                except:
                    CoT = response
                    Final_Answer = 'error'
                    parse_error = True
                    parse_error_attempt += 1
                    if llm_config['llm_type'] == 'ollama':
                        Final_Answer = None
                        parse_error = False                                            
                instruction_violation_count.append((attempts, parse_error_attempt))
            row[f'CoT_{round}'] = CoT
            row[f'Final Answer_{round}'] = Final_Answer
            row[f'Instruction Violation_{round}'] = instruction_violation_count

        save_csv(row, llm_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GSM8K evaluation')
    parser.add_argument('--llm_type', default='openai', choices=['openai', 'ollama', 'anthropic'], help='LLM type')
    parser.add_argument('--api_key_file', default='api_key_gy.txt', help='API key file')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name')
    parser.add_argument('--prompt_file', default='least_to_most.json', help='Prompt template file')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for LLM')
    parser.add_argument('--start_index', type=int, default=12, help='Starting index for processing')
    
    args = parser.parse_args()
    
    llm_config = get_llm_config(args)
    
    process_difficulty(llm_config, args.start_index)
