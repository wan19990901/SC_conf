import argparse
from LLM_agent import *
import os
import pandas as pd
from tqdm import tqdm

def get_llm_config(args):
    """
    Create LLM configuration dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict: Configuration for LLM including model type, API key, prompt template, etc.
    """
    return {
        'llm_type': args.llm_type,
        'api_key_link': args.api_key_file,
        'model': args.model,
        'prompt_link': args.prompt_file,
        'parser_template': CoT,
        'temperature': args.temperature,
    }

def save_results_to_csv(result_row, llm_config):
    """
    Save evaluation results to CSV file, creating directory if needed.
    
    Args:
        result_row: Pandas Series containing evaluation results
        llm_config: LLM configuration dictionary
    """
    output_dir = f'../data/Evaluation_CoTs/{llm_config["model"]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f'{args.dataset_name}.csv')
    
    # Add prompt file name to the results
    result_row['Prompt_File'] = os.path.basename(llm_config['prompt_link'])
    
    if not os.path.isfile(output_file):
        # Create new file with header
        result_row.to_frame().T.to_csv(output_file, mode='a', index=False, header=True)
    else:
        # Append to existing file
        result_row.to_frame().T.to_csv(output_file, mode='a', index=False, header=False)

def evaluate_questions(llm_config, start_index=0):
    """
    Process and evaluate questions using the specified LLM configuration.
    
    Args:
        llm_config: LLM configuration dictionary
        start_index: Index to start processing from (default: 0)
    """
    # Read API key
    with open(llm_config['api_key_link'], 'r') as f:
        api_key = f.read()
    
    # Load and subset the dataset
    dataset = pd.read_csv(os.path.join(args.data_dir, f'{args.dataset_name}.csv'))
    evaluation_subset = dataset[:args.num_samples]
    
    # Process each question
    for row_idx in tqdm(range(start_index, len(evaluation_subset)), colour='blue', desc='Processing Questions', position=0):
        current_row = evaluation_subset.iloc[row_idx]
        subject = current_row['Category']
        question = current_row['Question']
        
        input_args = {
            'subject': subject,
            'question': question
        }
        
        # Generate multiple chain-of-thought responses
        for attempt in tqdm(range(args.num_cot), colour='green', desc='Generating CoT', position=1):
            instruction_violations = []
            parsing_failed = True
            parse_attempts = 0
            
            # Retry parsing up to 3 times if needed
            while parsing_failed and parse_attempts < 3:
                # Initialize LLM agent
                cot_agent = LLM_agent(
                    llm_type=llm_config['llm_type'],
                    api_key=api_key,
                    model=llm_config['model'],
                    temperature=llm_config['temperature']
                )
                
                # Set prompt and parser
                cot_agent.set_prompt(llm_config['prompt_link'])
                if llm_config['llm_type'] != 'ollama':
                    cot_agent.set_parser(llm_config['parser_template'])
                
                # Generate response
                response, attempts = cot_agent.involk(input_args)
                if llm_config['llm_type'] == 'ollama':
                    response = response.content
                
                # Parse response
                try:
                    chain_of_thought = response['CoT']
                    final_answer = response['Final_answer']
                    parsing_failed = False
                except:
                    chain_of_thought = response
                    final_answer = 'error'
                    parsing_failed = True
                    parse_attempts += 1
                    if llm_config['llm_type'] == 'ollama':
                        final_answer = None
                        parsing_failed = False
                
                instruction_violations.append((attempts, parse_attempts))
            
            # Store results
            current_row[f'CoT_{attempt}'] = chain_of_thought
            current_row[f'Final Answer_{attempt}'] = final_answer
            current_row[f'Instruction Violation_{attempt}'] = instruction_violations
        
        save_results_to_csv(current_row, llm_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LLM performance on question answering tasks')
    
    # Data configuration
    parser.add_argument('--data_dir', default='../data/question_data/preprocessed/',
                      help='Directory containing the question datasets')
    parser.add_argument('--dataset_name', default='GSM8K',
                      help='Name of the dataset to evaluate')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of questions to evaluate (default: 100)')
    parser.add_argument('--num_cot', type=int, default=40,
                      help='Number of chain-of-thought generations per question (default: 40)')
    
    # LLM configuration
    parser.add_argument('--llm_type', default='openai',
                      choices=['openai', 'ollama', 'anthropic'],
                      help='Type of LLM to use')
    parser.add_argument('--api_key_file', default='api_key_gy.txt',
                      help='File containing API key')
    parser.add_argument('--model', default='gpt-4o-mini',
                      help='Name of the model to use')
    parser.add_argument('--prompt_file', default='zero_CoT.json',
                      help='File containing prompt template')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Temperature parameter for LLM')
    parser.add_argument('--start_index', type=int, default=12,
                      help='Starting index for processing questions')
    
    args = parser.parse_args()
    
    # Initialize and run evaluation
    llm_config = get_llm_config(args)
    evaluate_questions(llm_config, args.start_index)
