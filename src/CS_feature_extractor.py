import json
import os
import pandas as pd
import numpy as np
import re
import ast
from tqdm import tqdm
from utils import *
from collections import Counter
import statistics
import sys
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')


def count_steps(text):
    return len(re.findall(r'Step \d+:', str(text)))

def average_step_length(text):
    steps = re.split(r'Step \d+:', str(text))[1:]
    step_lengths = [len(word_tokenize(step)) for step in steps]
    return sum(step_lengths) / len(step_lengths) if step_lengths else 0

def mathematical_term_density(text):
    words = word_tokenize(str(text).lower())
    math_terms = ['calculate', 'number', 'integer', 'digit', 'product', 'divide', 'probability']
    math_term_count = sum(1 for word in words if word in math_terms)
    return math_term_count / len(words) if words else 0

def imperative_density(text):
    words = word_tokenize(str(text).lower())
    imperative_verbs = ['calculate', 'find', 'determine', 'compute', 'divide', 'multiply', 'add', 'subtract', 'output']
    imperative_count = sum(1 for word in words if word in imperative_verbs)
    return imperative_count / len(words) if words else 0

def consecutive_step_coherence(text):
    steps = re.split(r'Step \d+:', str(text))[1:]
    coherence_scores = []
    for i in range(len(steps) - 1):
        step1_words = set(word_tokenize(steps[i].lower()))
        step2_words = set(word_tokenize(steps[i+1].lower()))
        common_words = step1_words.intersection(step2_words)
        coherence_scores.append(len(common_words) / len(step1_words.union(step2_words)))
    return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def extract_sim(df,method = 'bigram',emb='jaccard'):
    if method == 'bigram':
        return calculate_similarity_with_bigram(df,method=emb)
    elif method == 'agg':
        return calculate_similarity_with_aggregation(df,method=emb)
    elif method == 'pw':
        return calculate_similarity_pairwise(df,method=emb)
    else:
        print('error')


def modify_float(num):
    try:
        return str(round(float(num), 1))
    except:
        return num


def extract_cot_answer(df):
    tmp = []
    binary_correct = []
    for col in df.columns:
        if col.startswith('Final Answer_'):
            final_ans = df[col].apply(modify_float)
            correct_ans = [modify_float(i) for i in df['Correct Answer']]
            correctness = 1*(final_ans == correct_ans)
            tmp.append(final_ans)
            binary_correct.append(correctness)
    tmp_arr = np.vstack(tmp)
    binary_arr = np.vstack(binary_correct).T
    cot_answer_arr = tmp_arr.T
    return cot_answer_arr,binary_arr

def extract_sim_input(df, method='jaccard'):
    # Get the 'Question' column as a list
    questions = df['Question'].tolist()
    
    # Initialize a list to store the similarity scores
    similarity_scores = []
    
    # Iterate over each column starting with 'CoT_'
    for col in df:
        if col.startswith('CoT_'):
            # Get the column values as a list
            cot_values = df[col].tolist()
            # Calculate the similarity between each 'Question' and 'CoT_' pair
            similarities = [calculate_similarity(method, question, cot) for question, cot in zip(questions, cot_values)]
            # Append the similarity scores to the list
            similarity_scores.append(similarities)
    
    # Convert the similarity scores to a NumPy array and transpose it
    similarity_scores = np.array(similarity_scores).T
    
    return similarity_scores

def extract_len(df):
    step_count_buffer = []
    for col in df:
        if col.startswith('CoT_'):
            cleaned_answers = []
            for entry in df[col]:
                entry_str = str(entry)
                # Look for patterns like Step 1: ... Step2:
                steps = re.findall(r'Step (\d+):', entry_str)
                
                if steps:
                    # If the pattern is found, use the last digit as the length
                    last_step = int(steps[-1])>4
                    cleaned_answers.append(int(last_step))
                else:
                    # If the pattern is not found, use nltk to tokenize the text into sentences
                    sentences = nltk.sent_tokenize(entry_str)
                    num_of_sentences = len(sentences)
                    cleaned_answers.append(int(max(num_of_sentences - 2, 0)>5))
            
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
                misktake = re.findall(r'(a mistake)|(an error)|(not solvable)|(not enough information)|(apologize)', str(entry))
                if misktake:
                    cleaned_answers.append(1)
                else:
                    # If no number is found, replace with 'error'
                    cleaned_answers.append(0)
            mistake_buffer.append(cleaned_answers)

    mistakes = np.array(mistake_buffer).T
    return mistakes


def extract_IV(df):
    instruction_buffer = []
    for col in df:
        if col.startswith('Final Answer_'):
            cleaned_answers = []
            for entry in df[col]:
                # Extract numbers using regular expression
                cleaned_answers.append(1 if str(entry).lower() == 'error' else 0)
            instruction_buffer.append(cleaned_answers)

    instruction_error = np.array(instruction_buffer).T
    return instruction_error

def extract_AC(arr, method='bigram'):
    consistency_checks = np.full(arr.shape, 0, dtype=int)
    if method == 'bigram':
        consistency_checks[:, 1:] = 1 * (arr[:, 1:] == arr[:, :-1])
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
    elif method == 'pw':
        results = []
        for row in arr:
            row_results = [0]  # The first element has no predecessors, initialize with 0.
            for i in range(1, len(row)):
                # Perform pairwise comparison between current item and all previous items
                comparisons = [row[i] == row[j] for j in range(i)]
                # Calculate the mode of the comparison results
                most_common_comparison = statistics.multimode(comparisons)
                # The multimode() function returns a list of the most frequent values.
                # We assign 1 if True is in the list, otherwise 0.
                comparison_mode = 1 if True in most_common_comparison else 0
                row_results.append(comparison_mode)
            results.append(row_results)
        consistency_checks = np.array(results)
    return consistency_checks

def extract_feature(df, features_li):
    # Initialize an empty dictionary for collecting features
    feature_dict = {
        'id': []
    }
    
    # Always include 'Name' and 'Model' if they are in the DataFrame
    always_include = ['Name', 'Model']
    for key in always_include:
        if key in df.columns:
            feature_dict[key] = []

    # Include 'correct answer', 'CoT answers', and 'Correctness' directly from function output
    feature_dict['correct answer'] = []
    feature_dict['CoT answers'] = []
    feature_dict['Correctness'] = []

    # Initialize lists in the dictionary only for requested features that are not already included
    requested_features = set(features_li) - set(always_include) - {'correct answer', 'CoT answers', 'Correctness'}
    for feature in requested_features:
        feature_dict[feature] = []

    cot_answer_arr, binary_arr = extract_cot_answer(df)
    # Populate 'CoT answers' and 'Correctness'
    for idx in range(len(df)):
        feature_dict['CoT answers'].append(cot_answer_arr[idx].tolist())
        feature_dict['Correctness'].append(binary_arr[idx].tolist())
    
    if 'DIF_IV' in features_li:
        IV = extract_IV(df)
    
    if 'LEN' in features_li:
        LEN = extract_len(df)
    
    if 'QUA_IM' in features_li:
        IM = extract_IM(df)
    
    sim_features = ['SIM_COT_BIGRAM', 'SIM_COT_AGG', 'SIM_AC_BIGRAM', 'SIM_AC_AGG', 'SIM_INPUT']
    sim_methods = {}
    for feature in sim_features:
        if feature in features_li:
            if feature == 'SIM_INPUT':
                sim_methods[feature] = extract_sim_input(df)
            else:
                method = feature.split('_')[-1].lower()
                if 'SIM_COT_' in feature:
                    sim_methods[feature] = extract_sim(df, method=method)
                elif 'SIM_AC_' in feature:
                    sim_methods[feature] = extract_AC(cot_answer_arr, method=method)

    # Populate feature data for each row
    for row in tqdm(range(len(df))):
        feature_dict['id'].append(row)
        for key in always_include:
            if key in feature_dict:
                feature_dict[key].append(df.iloc[row].get(key, None))
        
        if 'Correct Answer' in df.columns:
            feature_dict['correct answer'].append(df.iloc[row]['Correct Answer'])

        for feature in requested_features:
            if feature in sim_methods:
                feature_dict[feature].append(sim_methods[feature][row].tolist())
            elif feature == 'DIF_IV':
                feature_dict[feature].append(IV[row].tolist())
            elif feature == 'LEN':
                feature_dict[feature].append(LEN[row].tolist())
            elif feature == 'QUA_IM':
                feature_dict[feature].append(IM[row].tolist())
            elif feature == 'STEP_COUNT':
                feature_dict[feature].append([count_steps(cot) for cot in df.iloc[row][[col for col in df.columns if col.startswith('CoT_')]]])
            elif feature == 'AVG_STEP_LENGTH':
                feature_dict[feature].append([average_step_length(cot) for cot in df.iloc[row][[col for col in df.columns if col.startswith('CoT_')]]])
            elif feature == 'MATH_TERM_DENSITY':
                feature_dict[feature].append([mathematical_term_density(cot) for cot in df.iloc[row][[col for col in df.columns if col.startswith('CoT_')]]])
            elif feature == 'IMPERATIVE_DENSITY':
                feature_dict[feature].append([imperative_density(cot) for cot in df.iloc[row][[col for col in df.columns if col.startswith('CoT_')]]])
            elif feature == 'STEP_COHERENCE':
                feature_dict[feature].append([consecutive_step_coherence(cot) for cot in df.iloc[row][[col for col in df.columns if col.startswith('CoT_')]]])

    # Convert dictionary to DataFrame
    return feature_dict

if __name__ == '__main__':
    
    # DATA_DIR = "../data/adaptive_consistency_outputs/"
    # input_file_path = os.path.join(DATA_DIR, 'final_asc.csv')
    DATA_DIR = "../data/CoT_data/"
    input_file_path = os.path.join(DATA_DIR, 'final_train.csv')
    df = pd.read_csv(input_file_path).sample(frac=0.5).reset_index(drop=True)
    print(df.shape)
    feature_li = ['LEN', 'QUA_IM', 'DIF_IV', 'SIM_COT_BIGRAM', 'SIM_COT_AGG', 'SIM_AC_BIGRAM', 'SIM_AC_AGG', 'SIM_INPUT', 'STEP_COUNT', 'AVG_STEP_LENGTH', 'MATH_TERM_DENSITY', 'IMPERATIVE_DENSITY', 'STEP_COHERENCE']
    data = extract_feature(df,feature_li)
    df_to_save = pd.DataFrame(data)

    df = calculate_SC_correctness(df_to_save)
    
    # Calculate Early Stopping Correctness with a specific window size
    window_size = 5  # Define your window size
    df = calculate_ES_correctness(df, window_size)
    
    # Calculate Adaptive Consensus Correctness
    df = calculate_ASC_correctness(df)

    storage_dir = os.path.join(DATA_DIR, 'new_extracted_data')
    os.makedirs(storage_dir, exist_ok=True)

    # Save df_to_save before prepare_df
    output_file_name_before = 'final_extracted.json'
    file_store_path_before = os.path.join(storage_dir, output_file_name_before)
    df_to_save.to_json(file_store_path_before, orient='records', lines=True)
    print(f'File saved in : {file_store_path_before}')

