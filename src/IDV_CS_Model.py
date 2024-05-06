import os
import pandas as pd
from CS_feature_extractor import extract_feature
from collections import Counter
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score,roc_curve,f1_score
from sklearn.linear_model import LogisticRegression
from adaptive_consistency import AC, BetaStoppingCriteria

DATA_DIR = '../data'
DF_NAME = 'GSM8K'
DIFFICULTY = 'easy'
NUM_OF_SAMPLES = 500
NUM_OF_COT = 40
MODEL = 'gpt-3.5-turbo-0125'
def calculate_SC_correctness(df):
    # Define a helper function to determine the majority and compare it with the correct answer
    def check_majority(answers, correct):
        if not answers:
            return 0
        # Count the occurrences of each answer and find the most common one
        most_common = Counter(answers).most_common(1)[0][0]
        # Compare the most common answer with the correct answer
        return 1 if most_common == str(correct) else 0

    # Apply the helper function across the DataFrame rows
    df['SC_correctness'] = df.apply(lambda row: check_majority(row['CoT answers'], row['correct answer']), axis=1)
    return df


def calculate_ES_correctness(df, window_size):
    # Helper function to evaluate answers with a sliding window
    def evaluate_window(answers, correct):
        # Track the number of steps (checks) taken
        steps = window_size - 1

        # If the list is shorter than the window size, adjust the window size to the length of the list
        window_size_adjusted = min(window_size, len(answers))

        # Slide through the list with the adjusted window size
        for i in range(len(answers) - window_size_adjusted + 1):
            # Check the window content
            window = answers[i:i + window_size_adjusted]
            steps += 1

            # If all elements in the window are the same, evaluate correctness
            if window.count(window[0]) == window_size_adjusted:
                return 1 if window[0] == str(correct) else 0, steps

        # If no unanimous window is found, use the same value as SC_correctness and set steps to the length of the list
        majority = Counter(answers).most_common(1)[0][0]
        correctness = 1 if majority == str(correct) else 0
        return correctness, len(answers)

    # Apply the function to each row in the DataFrame
    result = df.apply(lambda row: evaluate_window(row['CoT answers'], row['correct answer']), axis=1)
    df['ES_correctness'] = result.apply(lambda x: x[0])
    df['ES_steps'] = result.apply(lambda x: x[1])

    return df


def concatenate_columns(df, data_columns, outcome_column):
    # Initialize an empty dictionary to store the concatenated data
    concatenated_data = {}

    # Get the number of rows based on the length of the outcome column
    num_rows = len(df)

    # Iterate over each column
    for column in data_columns + [outcome_column]:
        # Initialize an empty list to store the concatenated values for the current column
        concatenated_values = []

        # Iterate over each row
        for i in range(num_rows):
            # Get the list of values for the current column and row

            values = df[column].iloc[i]

            # Concatenate the values into a single string
            concatenated_values += list(values)

        # Add the concatenated values to the dictionary with the column name as the key
        concatenated_data[column] = concatenated_values

    # Add the outcome column to the concatenated data dictionary

    # Create a DataFrame from the concatenated data dictionary
    df = pd.DataFrame(concatenated_data)

    return df
def calculate_ASC_correctness(df):
    # Define the helper function to get majority vote and length of answers
    def majority_and_length(answers):
        if not answers:
            return None, 0
        most_common_answer = Counter(answers).most_common(1)[0][0]
        return most_common_answer, len(answers)

    # Prepare to collect data for new columns
    asc_correctness = []
    asc_steps = []
    ac = AC(stop_criteria=BetaStoppingCriteria(0.95), max_gens=40)

    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        answers = row['CoT answers']
        correct_answer = row['correct answer']
        # Assuming ac.should_stop is a method that determines when to stop appending answers
        # Implement your condition in ac.should_stop(answers) or define it accordingly
        for i, answer in enumerate(answers):
            if ac.should_stop(answers[:i+1]):  # Pass the slice up to the current point
                break
        # Get the majority vote and length of the sequence
        majority_vote, length_of_answers = majority_and_length(answers[:i+1])
        # Compare majority vote to the correct answer and determine correctness
        asc_correctness.append(1 if majority_vote == str(correct_answer) else 0)
        asc_steps.append(length_of_answers)

    # Update DataFrame with new columns
    df['asc_correctness'] = asc_correctness
    df['asc_steps'] = asc_steps

    return df

# Example of integrating the new function into your existing workflow
def prepare_df(df, feature_li, ES_window_size = 5):
    # Reset first sim to 0.5
    for row_idx in range(len(df)):
        df['SIM_COT_AGG'][row_idx][0] = 0.5

    # Adding Self Consistency Baseline
    df = calculate_SC_correctness(df)
    print(df.SC_correctness.value_counts())

    # Adding Early Stopping Self Consistency Baseline
    df = calculate_ES_correctness(df, window_size=ES_window_size)
    print(df.ES_correctness.value_counts())

    # Adding ASC Correctness and Steps
    print(df.columns)
    df = calculate_ASC_correctness(df)
    print("ASC Steps and Answers added.")

    # Concate cols
    df_concate = concatenate_columns(df, feature_li, 'Correctness')
    df_concate['cot_answer'] = np.repeat(df['correct answer'].values, NUM_OF_COT)
    print('------------DF Stats-------------')
    for col in df_concate.columns:
        if not col.startswith('SIM_COT_'):
            print(col, ':', Counter(df_concate[col]))
    return df_concate
def customized_LR(coe,intercept,features):
    lincomb = sum([coe[i]*features[i] for i in range(len(coe))]) +intercept
    return 1 / (1 + np.exp(-lincomb))
def customized_LR_model(df,feature_li,coe,intercept):

    df_concate = prepare_df(df,feature_li)
    split_idx = int(len(df_concate) * 0.8)

    X_test_hard = df_concate[feature_li].iloc[split_idx:]
    y_test_hard = df_concate['Correctness'].iloc[split_idx:]

    y_pred_proba = X_test_hard.apply(
        lambda row: customized_LR(
            coe=coe,  # Flatten the coef_ array if it's 2D
            intercept=intercept,
            features=row
        ),
        axis=1  # Apply the function to each row
    )
    print("Coefficients:", coe)
    print("Intercept:", intercept)
    auroc = roc_auc_score(y_test_hard, y_pred_proba)
    print(f"The AUROC score is: {auroc}")
    df_concate['confidence_score'] = df_concate[feature_li].apply(
        lambda row: customized_LR(
            coe=coe,  # Flatten the coef_ array if it's 2D
            intercept=intercept,
            features=row
        ),
        axis=1  # Apply the function to each row
    )
    lists = [df_concate['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_concate), NUM_OF_COT)]
    df['confidence_score'] = lists
    print('=================================================================================')
    return df,0.36

def trained_LR_model(df,feature_li):
    df_concate = prepare_df(df, feature_li)
    split_idx = int(len(df_concate) * 0.8)  # 80% of the length of the dataset

    # Split the data into training and test sets
    X_train_hard = df_concate[feature_li].iloc[:split_idx]
    y_train_hard = df_concate['Correctness'].iloc[:split_idx]
    X_test_hard = df_concate[feature_li].iloc[split_idx:]
    y_test_hard = df_concate['Correctness'].iloc[split_idx:]

    # Add a constant term to the features for the intercept for training and testing set
    X_train_hard = sm.add_constant(X_train_hard)
    X_test_hard = sm.add_constant(X_test_hard)

    # Fit the logistic regression model using statsmodels
    model = sm.Logit(y_train_hard, X_train_hard)
    result = model.fit()
    print(result.summary())

    # Split the data into training and test sets
    X_train_hard = df_concate[feature_li].iloc[:split_idx]
    y_train_hard = df_concate['Correctness'].iloc[:split_idx]
    X_test_hard = df_concate[feature_li].iloc[split_idx:]
    y_test_hard = df_concate['Correctness'].iloc[split_idx:]

    # Initialize and fit the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_hard, y_train_hard)

    # Print the model coefficients and intercept
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Make predictions on the test data (predicting probabilities)
    y_pred_proba = model.predict_proba(X_test_hard)[:, 1]  # Get probabilities for the positive class

    # Calculate the AUROC
    auroc = roc_auc_score(y_test_hard, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test_hard, y_pred_proba)
    f1_scores = [f1_score(y_test_hard, y_pred_proba > thresh) for thresh in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"The AUROC score is: {auroc}")

    df_concate['confidence_score'] = model.predict_proba(df_concate[feature_li])[:, 1]
    lists = [df_concate['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_concate), NUM_OF_COT)]
    df['confidence_score'] = lists
    print('=================================================================================')
    return df,best_threshold


if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'final.csv')
    df_raw = pd.read_csv(file_path)
    df_with_features = pd.DataFrame(extract_feature(df_raw))
    feature_li = [
        'LEN',
        'QUA_IM',
        'DIF_IV',
        # 'DIF_SUB',
        # 'SIM_COT_BIGRAM',
        'SIM_COT_AGG',
        # 'SIM_COT_PW',
        'SIM_AC_BIGRAM',
        'SIM_AC_AGG',
        'SIM_AC_PW',
        # 'size_of_cot'
    ]
    # coe = [-0.1,-5,-1,3,2,2,2]
    # intercept = -1.5
    df = trained_LR_model(df_with_features,feature_li)