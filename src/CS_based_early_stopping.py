from IDV_CS_Model import *
import sys

DATA_DIR = '../data/Evaluation_CoTs/'
# DF_NAME = 'GSM8K'
# DIFFICULTY = 'easy'
# NUM_OF_SAMPLES = 500
# NUM_OF_COT = 40
# MODEL = 'gpt-3.5-turbo-0125'


def normalize_cs(cs_li, threshold):
    cs_arr = np.array(cs_li)
    # normalized_cs = [(cs-threshold)/(1-threshold) if cs > threshold else (cs-threshold)/(threshold) for cs in cs_arr]
    normalized_cs = cs_arr - threshold
    # normalized_cs = [0.5 if cs > threshold else -0.5 for cs in cs_arr]
    return np.array(normalized_cs)


def stop_con1(individual_cs):
    cumulative_difference = (individual_cs).cumsum()
    stop_idx = np.argmax(cumulative_difference > 0.5)
    return stop_idx


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
    CS_Answer = []
    CS_correctness = []
    CS_steps = []
    if stop_mechanism == 'PositiveN':
        for row_idx in range(len(df)):
            test_row = df.iloc[row_idx]
            individual_cs = normalize_cs(test_row['confidence_score'], threshold)
            # individual_cs = test_row['confidence_score'][warm_up_steps:] - threshold
            stop_idx = stop_con2(individual_cs, buffer_size=N)

            if stop_idx:
                num_of_steps = stop_idx
            else:
                num_of_steps = 40
            answers = test_row['CoT answers'][:num_of_steps]
            scores = individual_cs[:num_of_steps]
            weighted_votes = Counter()
            for answer, score in zip(answers, scores):
                if score > 0:
                    weighted_votes[answer] += score
            # Find the answer with the highest total score
            if (len(weighted_votes)) == 0:
                for answer, score in zip(answers, scores):
                    weighted_votes[answer] += score
            result = max(weighted_votes, key=weighted_votes.get)
            CS_Answer.append(result)
            CS_correctness.append(1 if result == str(test_row['correct answer']) else 0)
            CS_steps.append(num_of_steps)
    elif stop_mechanism == 'ConsistencyN':
        found_count = 0
        for idx, row in df.iterrows():
            confidence_scores = row['confidence_score']
            answers = row['CoT answers']
            found, num_of_steps, answer = consecutive_scores_above_threshold(confidence_scores, answers, threshold, N)
            if found:
                found_count += 1
                CS_Answer.append(answer)
                CS_correctness.append(1 if answer == str(row['correct answer']) else 0)
                CS_steps.append(num_of_steps)  # +1 to account for the 0-indexing
            else:
                CS_Answer.append(None)
                CS_correctness.append(row['SC_correctness'])
                CS_steps.append(num_of_steps)
    df['CS_Answer'] = CS_Answer
    df['CS_correctness'] = CS_correctness
    df['CS_steps'] = CS_steps

    df_model_comp_dict = {}
    df_model_comp_dict['SC_ACC'] = df.SC_correctness.sum() / len(df)
    df_model_comp_dict['ES_ACC'] = df.ES_correctness.sum() / len(df)
    df_model_comp_dict['CS_ACC'] = df.CS_correctness.sum() / len(df)
    df_model_comp_dict['SC_Avg_Steps'] = 40
    df_model_comp_dict['ES_Avg_Steps'] = df.ES_steps.mean()
    df_model_comp_dict['CS_Avg_Steps'] = df.CS_steps.mean()

    for key, val in df_model_comp_dict.items():
        print(key, ' : ', val)

    return df, df_model_comp_dict


if __name__ == '__main__':
    # storage_dir = os.path.join(DATA_DIR, f'Evaluation_CoTs/{MODEL}')
    file_path = os.path.join(DATA_DIR, 'final.csv')
    df_raw = pd.read_csv(file_path)
    df_raw = df_raw.sample(1000).reset_index(drop=True)
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
    # coe = [-0.1, -5, -1, 3, 2, 2, 2]
    # intercept = -2
    # df_cs, threshold = customized_LR_model(df=df_with_features, feature_li=feature_li, coe=coe, intercept=intercept)
    # threshold = float(sys.argv[1])
    # N = int(sys.argv[2])
    N = 3
    threshold = 0.5
    df_cs, _ = trained_LR_model(df=df_with_features, feature_li=feature_li)
    stop_mechanism = 'ConsistencyN'
    df, _ = CS_early_stopping(df=df_cs, threshold=threshold, N=N, stop_mechanism=stop_mechanism)
    file_name = f"df_threshold_{threshold}_N_{N}_stop_{stop_mechanism}.csv"
    storage_dir = '../result/experiments_output/'
    df.to_csv(storage_dir + file_name, index=False)
