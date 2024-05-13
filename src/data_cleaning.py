import os
import re
import pandas as pd

def load_data(main_dir, subdirectories):
    dataframes = []
    for subdir in subdirectories:
        for file in os.listdir(os.path.join(main_dir, subdir)):
            if file.endswith('.csv'):
                difficulty = 'easy' if 'easy' in file else 'hard'
                df = pd.read_csv(os.path.join(main_dir, subdir, file))
                df['Difficulty'] = difficulty
                df['Model'] = subdir
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def preprocess_data(df):
    unnamed_cols = ['index'] + [col for col in df.columns if col.startswith('Unnamed:')]
    df.drop(columns=unnamed_cols, inplace=True)
    df.dropna(inplace=True)
    df = df[df['Correct Answer'] != 'NO']
    return df

def clean_numeric_value(value):
    digit_re = re.compile(r"-?\$?\d+\.?\d*%?")
    value = str(value).replace('$', '').replace('%', '')
    matches = digit_re.findall(value)
    if matches:
        num = matches[0].rstrip('%')
        cleaned_value = str(float(num))
    else:
        cleaned_value = "error"
    return cleaned_value

def clean_numeric_columns(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_numeric_value)

def clean_gsm8k_answers(df):
    answer_columns = [f"Final Answer_{i}" for i in range(40)] + ["Correct Answer"]
    clean_numeric_columns(df, answer_columns)

def extract_options(question):
    options_text = question.split('Options:\n')[-1]
    options = re.findall(r"\((.)\)\s(.+)", options_text)
    return {opt[1].strip(): opt[0] for opt in options}

def clean_bb_answer(answer, options):
    answer = answer.strip()
    if len(answer) == 1 and answer.upper() in options.values():
        return answer.upper()
    elif answer in options:
        return options[answer]
    else:
        return "error"

def process_bb_answers(df):
    for i in range(40):
        answer_column = f"Final Answer_{i}"
        df[answer_column] = df.apply(lambda x: clean_bb_answer(x[answer_column], extract_options(x['Question'])), axis=1)

def extract_math_options(question):
    options_text = question.split('The options are:')[-1]
    options = re.findall(r"(\w)\s\)\s(\d+)", options_text)
    return {opt[1].strip(): opt[0].lower() for opt in options}

def clean_math_answer(answer, options):
    answer = answer.strip()
    if answer in options:
        return options[answer]
    elif len(answer) == 1 and answer.lower() in options.values():
        return answer.lower()
    else:
        return "error"

def process_math_answers(df):
    for i in range(40):
        answer_column = f"Final Answer_{i}"
        df[answer_column] = df.apply(lambda x: clean_math_answer(x[answer_column], extract_math_options(x['Question'])), axis=1)

def main():
    main_dir = "../data/Evaluation_CoTs"
    subdirectories = ["claude-3-haiku-20240307", "gpt-3.5-turbo-0125", "gpt-4"]

    final_df = load_data(main_dir, subdirectories)
    final_df = preprocess_data(final_df)

    df_gsm8k = final_df[final_df.Name.str.startswith('GSM')]
    clean_gsm8k_answers(df_gsm8k)

    df_bb = final_df[final_df.Name.str.startswith('Big')]
    process_bb_answers(df_bb)

    df_math = final_df[final_df.Name.str.startswith('Math')]
    process_math_answers(df_math)

    pd.concat([df_gsm8k, df_bb, df_math]).reset_index(drop=True).to_csv('../data/final.csv', index=False)

if __name__ == "__main__":
    main()