import json
import os
import pandas as pd
import re
import ast
from tqdm import tqdm
from utils import *
from collections import Counter
from scipy.stats import mode
DATA_DIR = '../data'

# Experiment Config
DF_NAME = 'MathQA'
DIFFICULTY = 'easy'
NUM_OF_SAMPLES = 500
NUM_OF_COT = 40
MODEL = 'gpt-3.5-turbo-0125'

def extract_sim(df,method = 'bigram',emb='jaccard'):
    if method == 'bigram':
        return calculate_similarity_with_bigram(df,method=emb)
    elif method == 'agg':
        return calculate_similarity_with_aggregation(df,method=emb)
    elif method == 'pw':
        return calculate_similarity_pairwise(df,method=emb)
    else:
        print('error')
def extract_cot_answer(df):
    tmp = []
    binary_correct = []
    for col in df.columns:
        if col.startswith('Final Answer_'):
            final_ans=df[col].to_numpy()
            correct_ans = [str(i) for i in df['Correct Answer']]
            correctness = 1*(final_ans == correct_ans)
            tmp.append(final_ans)
            binary_correct.append(correctness)
    tmp_arr = np.vstack(tmp)
    binary_arr = np.vstack(binary_correct).T
    cot_answer_arr = tmp_arr.T
    return cot_answer_arr,binary_arr

def extract_len(df):
    step_count_buffer = []
    for col in df:
        if col.startswith('CoT_'):
            cleaned_answers = []
            for entry in df[col]:
                # Extract numbers using regular expression
                steps = re.findall(r'[Ss]tep\s?\d', str(entry))
                if steps:
                    # Join all numbers with space if there are multiple numbers
                    cleaned_answers.append(len(steps))
                else:
                    # If no number is found, replace with 'error'
                    cleaned_answers.append(0)
            step_count_buffer.append(cleaned_answers)

    step_count = np.array(step_count_buffer).T
    return step_count
def extract_IM(df):
    mistake_buffer = []
    for col in df:
        if col.startswith('CoT_'):
            cleaned_answers = []
            for entry in df[col]:
                # Extract numbers using regular expression
                misktake = re.findall(r'(be a mistake)|(be an error)', str(entry))
                if misktake:
                    cleaned_answers.append(1)
                else:
                    # If no number is found, replace with 'error'
                    cleaned_answers.append(0)
            mistake_buffer.append(cleaned_answers)

    mistakes = np.array(mistake_buffer).T
    return mistakes

# def extract_UKW(df):
#     UKW_buffer = []
#     for col in df:
#         if col.startswith('CoT_'):
#             cleaned_answers = []
#             for entry in df[col]:
#                 # Extract numbers using regular expression
#                 UKW = re.findall(r'(be a mistake)|(be an error)', str(entry))
#                 if UKW:
#                     cleaned_answers.append(1)
#                 else:
#                     # If no number is found, replace with 'error'
#                     cleaned_answers.append(0)
#             UKW_buffer.append(cleaned_answers)
#
#     UKW_arr = np.array(UKW_buffer).T
#     return UKW_arr
def extract_IV(df):
    instruction_buffer = []
    for col in df:
        if col.startswith('Instruction Violation_'):
            cleaned_answers = []
            for entry in df[col]:
                # Extract numbers using regular expression
                x = ast.literal_eval(entry)
                cleaned_answers.append(sum([sum(idx) for idx in x]))
            instruction_buffer.append(cleaned_answers)

    instruction_error = np.array(instruction_buffer).T
    return instruction_error

def extract_AC(arr,method = 'bigram'):
    consistency_checks = np.full(arr.shape, 0, dtype=int)
    if method == 'bigram':
        consistency_checks[:, 1:] = 1*(arr[:, 1:] == arr[:, :-1])
    elif method == 'agg':
        results = []
        for row in arr:
            result = [0]  # Initialize with 0 for the first item.
            for i in range(1, len(row)):
                # Get the most common element in the array up to the current position
                most_common_item, _ = Counter(row[:i]).most_common(1)[0]
                # Append 1 if the current item matches the most common, else append 0
                result.append(1 if row[i] == most_common_item else 0)
            results.append(result)
        consistency_checks = np.array(results)
    elif method =='pw':
        results = []
        for row in arr:
            row_results = [0]  # The first element has no predecessors, initialize with 0.
            for i in range(1, len(row)):
                # Perform pairwise comparison between current item and all previous items
                comparisons = [row[i] == row[j] for j in range(i)]
                # Calculate the mode of the comparison results
                most_common_comparison, count = mode(comparisons)
                # The mode() function returns the smallest mode in case of multiple modes.
                # To handle this, we will consider 'True' as the mode if it's one of the modes and its count > 1.
                if True in most_common_comparison and count[0] > 1:
                    comparison_mode = 1
                else:
                    comparison_mode = 0
                row_results.append(comparison_mode)
            results.append(row_results)
        consistency_checks = np.array(results)
    return consistency_checks
def extract_feature(df):
    feature_dict = {
        'id': [],
        'correct answer': [],
        'CoT answers': [],
        'Correctness': [],
        'LEN': [],
        'QUA_IM': [],
        # ('QUA', 'UKW'): [],
        'DIF_IV': [],
        'DIF_SUB': [],
        'SIM_COT_BIGRAM': [],
        'SIM_COT_AGG': [],
        'SIM_COT_PW': [],
        'SIM_AC_BIGRAM': [],
        'SIM_AC_AGG': [],
        'SIM_AC_PW': [],
    }
    cot_answer_arr, binary_arr = extract_cot_answer(df)
    IV = extract_IV(df)
    LEN = extract_len(df)
    IM = extract_IM(df)
    SIM_cot_bigram = extract_sim(df, method='bigram')
    SIM_cot_agg = extract_sim(df, method='agg')
    SIM_cot_pw = extract_sim(df, method='pw')
    SIM_AC_bigram = extract_AC(cot_answer_arr, method='bigram')
    SIM_AC_agg = extract_AC(cot_answer_arr, method='agg')
    SIM_AC_pw = extract_AC(cot_answer_arr, method='pw')

    assert (cot_answer_arr).shape == (binary_arr).shape == (IV).shape == (LEN).shape == (SIM_cot_bigram).shape
    for row in tqdm(range(len(df))):
        feature_dict['id'].append(row)
        feature_dict['correct answer'].append(df.iloc[row]['Correct Answer'])
        feature_dict['DIF_SUB'].append(df.iloc[row]['Category'])
        feature_dict['CoT answers'].append(cot_answer_arr[row].tolist())
        feature_dict['Correctness'].append(binary_arr[row].tolist())
        feature_dict['QUA_IM'].append(IM[row].tolist())
        feature_dict['DIF_IV'].append(IV[row].tolist())
        feature_dict['LEN'].append(LEN[row].tolist())
        feature_dict['SIM_COT_BIGRAM'].append(SIM_cot_bigram[row].tolist())
        feature_dict['SIM_COT_AGG'].append(SIM_cot_agg[row].tolist())
        feature_dict['SIM_COT_PW'].append(SIM_cot_pw[row].tolist())
        feature_dict['SIM_AC_BIGRAM'].append(SIM_AC_bigram[row].tolist())
        feature_dict['SIM_AC_AGG'].append(SIM_AC_agg[row].tolist())
        feature_dict['SIM_AC_PW'].append(SIM_AC_pw[row].tolist())
    return feature_dict
if __name__ == '__main__':
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/{MODEL}')
    file_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.csv')
    df = pd.read_csv(file_path)
    data = extract_feature(df)

    df_to_save = pd.DataFrame(data)
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/Algo_Design_Data')

    file_store_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.json')
    with open(file_store_path,'w') as f:
        json.dump(data,f)
    # df_to_save.to_csv(file_store_path,index=False)

