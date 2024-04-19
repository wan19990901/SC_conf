import os
import pandas
import numpy as np
import pandas as pd
import re
import ast
from tqdm import tqdm
DATA_DIR = '../data'

# Experiment Config
DF_NAME = 'GSM8K'
DIFFICULTY = 'easy'
NUM_OF_SAMPLES = 500
NUM_OF_COT = 40
MODEL = 'gpt-3.5-turbo-0125'

def extract_cot_answer(df):
    tmp = []
    binary_correct = []
    for col in df.columns:
        if col.startswith('Final Answer_'):
            final_ans=df[col].to_numpy()
            correct_ans = [str(i) for i in df['Correct Answer']]
            correctness = final_ans == correct_ans
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

def extract_AC(arr):
    consistency_checks = np.full(arr.shape, False, dtype=bool)

    # Perform the check, starting from the second column (index 1)
    consistency_checks[:, 1:] = arr[:, 1:] == arr[:, :-1]
    return consistency_checks

if __name__ == '__main__':
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/{MODEL}')
    file_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.csv')
    df = pd.read_csv(file_path)
    feature_dict = {
        'id': [],
        'correct answer': [],
        'CoT answers': [],
        'Correctness': [],
        'LEN': [],
        ('QUA', 'IM'): [],
        # ('QUA', 'UKW'): [],
        ('DIF', 'IV'): [],
        ('DIF', 'SUB'): [],
        'AC': []
    }
    cot_answer_arr, binary_arr = extract_cot_answer(df)
    IV = extract_IV(df)
    LEN = extract_len(df)
    IM = extract_IM(df)
    AC = extract_AC(cot_answer_arr)
    assert (cot_answer_arr).shape == (binary_arr).shape == (IV).shape == (LEN).shape == (AC).shape
    for row in tqdm(range(len(df))):
        feature_dict['id'].append(row)
        feature_dict['correct answer'].append(df.iloc[row]['Correct Answer'])
        feature_dict[('DIF', 'SUB')].append(df.iloc[row]['Category'])
        feature_dict['CoT answers'].append(cot_answer_arr[row])
        feature_dict['Correctness'].append(binary_arr[row])
        feature_dict[('QUA', 'IM')].append(IM[row])
        feature_dict[('DIF', 'IV')].append(IV[row])
        feature_dict['LEN'].append(LEN[row])
        feature_dict['AC'].append(AC[row])

    df_to_save = pd.DataFrame(feature_dict)
    storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/Algo_Design_Data')
    file_store_path = os.path.join(storage_dir, f'{DF_NAME}_{DIFFICULTY}.csv')
    df_to_save.to_csv(file_store_path,index= False)

