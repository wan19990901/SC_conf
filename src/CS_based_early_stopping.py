from IDV_CS_Model import *
import sys
import time




def normalize_cs(cs_li, threshold):
    cs_arr = np.array(cs_li)
    normalized_cs = [(cs-threshold)/(1-threshold) if cs > threshold else (cs-threshold)/(threshold) for cs in cs_arr]
    # normalized_cs = cs_arr - threshold
    # normalized_cs = [0.5 if cs > threshold else -0.5 for cs in cs_arr]
    return np.array(normalized_cs)


def stop_con2(individual_cs, buffer_size=5):
    buffer = []
    for idx, cs in enumerate(individual_cs):
        if cs > 0:
            buffer.append(idx)
        if len(buffer) == buffer_size:
            return buffer[-1]


def consecutive_scores_above_threshold(scores, answers, threshold, n):
    for i in range(len(scores) - n + 1):
        # Check if all the next 'n' scores are above the threshold
        if all(score > threshold for score in scores[i:i + n]):
            # Check if all answers in this range are the same
            if len(set(answers[i:i + n])) == 1:
                return True, i + n, answers[i]
    return False, len(scores), None  # If no consecutive scores found or answers differ


def CS_early_stopping(df, threshold, N=5, stop_mechanism='PositiveN'):
    def compare_answers(answer1, answer2):
        try:
            return float(answer1) == float(answer2)
        except ValueError:
            return str(answer1).strip().lower() == str(answer2).strip().lower()

    CS_Answer = []
    CS_correctness = []
    CS_steps = []

    if stop_mechanism == 'PositiveN':
        for row_idx in range(len(df)):
            test_row = df.iloc[row_idx]
            individual_cs = normalize_cs(test_row['confidence_score'], threshold)
            stop_idx = stop_con2(individual_cs, buffer_size=N)

            num_of_steps = stop_idx + 1 if stop_idx else 40
            answers = test_row['CoT answers'][:num_of_steps]
            scores = individual_cs[:num_of_steps]
            weighted_votes = Counter()

            for answer, score in zip(answers, scores):
                if score > 0:
                    weighted_votes[answer] += score

            if len(weighted_votes) == 0:
                for answer, score in zip(answers, scores):
                    weighted_votes[answer] += score

            result = max(weighted_votes, key=weighted_votes.get)
            CS_Answer.append(result)
            CS_correctness.append(1 if compare_answers(result, test_row['correct answer']) else 0)
            CS_steps.append(num_of_steps)

    elif stop_mechanism == 'ConsistencyN':
        for idx, row in df.iterrows():
            confidence_scores = row['confidence_score']
            answers = row['CoT answers']
            found, num_of_steps, answer = consecutive_scores_above_threshold(confidence_scores, answers, threshold, N)

            if found:
                CS_Answer.append(answer)
                CS_correctness.append(1 if compare_answers(answer, row['correct answer']) else 0)
                CS_steps.append(num_of_steps)
            else:
                CS_Answer.append(None)
                CS_correctness.append(row['SC_correctness'])
                CS_steps.append(num_of_steps)
    df['CS_Answer'] = CS_Answer
    df['CS_correctness'] = CS_correctness
    df['CS_steps'] = CS_steps

    df_model_comp_dict = {
        'SC_ACC': df.SC_correctness.sum() / len(df),
        'ES_ACC': df.ES_correctness.sum() / len(df),
        'CS_ACC': df.CS_correctness.sum() / len(df),
        'SC_Avg_Steps': 40,
        'ES_Avg_Steps': df.ES_steps.mean(),
        'CS_Avg_Steps': df.CS_steps.mean(),
        'ASC_Avg_Steps': df.asc_steps.mean(),
        'ASC_ACC': df.asc_correctness.sum() / len(df)
    }
    # Print each metric
    for key, val in df_model_comp_dict.items():
        print(f"{key} : {val}")

    return df


if __name__ == '__main__':
    # Read JSON data
    DATA_DIR = '../data/CoT_data/new_extracted_data/'
    file_path = os.path.join(DATA_DIR, 'final_extracted_train.json')
    df_with_features = pd.read_json(file_path, lines=True)
    df_with_features = df_with_features[df_with_features.Model != 'gpt-4'].reset_index(drop=True)
    # Define the features list
    feature_li = ['LEN', 'QUA_IM', 'SIM_COT_BIGRAM', 'SIM_COT_AGG', 'SIM_AC_BIGRAM', 'SIM_AC_AGG', 'SIM_INPUT', 'STEP_COUNT', 'STEP_COHERENCE']

    # Continue with the rest of the script
    # coe = [0, -10, -2, 3, 1, 2]
    # intercept = -1
    # df_cs = customized_LR_model(df_with_features,feature_li,coe, intercept, report_auroc=True)
    df_cs = trained_LR_model(df_with_features, feature_li, report_auroc=False)

    N = int(sys.argv[2])
    threshold = float(sys.argv[1])

    # Applying early stopping mechanism
    df_final = CS_early_stopping(df=df_cs, threshold=threshold, N=N)

    # Saving the resulting DataFrame
    file_name = f"df_threshold_{threshold}_N_{N}.csv"
    storage_dir = '../result/experiments_output/test_N_threshold/'
    df_final.to_csv(os.path.join(storage_dir, file_name), index=False)
