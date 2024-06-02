import os
import pandas as pd


li_of_data = []
# List the names of directories in the current working directory
model_names = [name for name in os.listdir() if os.path.isdir(name)]
for model in model_names:
    benchmark_names = [name for name in os.listdir(model) if os.path.isdir(f'{model}/{name}')]
    for benchmark in benchmark_names:
        directory = f'{model}/{benchmark}'
        for file_name in os.listdir(directory):
            if file_name.endswith(f"_cleaned.csv"):
                file_path = os.path.join(directory, file_name)
                try:
                    df = pd.read_csv(file_path)
                except:
                    continue
                df['Name'] = [benchmark for i in range(len(df))]
                df['Model'] = [model.split('_')[0] for i in range(len(df))]
                li_of_data.append(df)
                print(f"File '{file_path}' has been read as a DataFrame.")
            else:
                print("No file ending with f'_cleaned.csv' found in the directory.")
df_all = pd.concat(li_of_data,ignore_index=True)
df_all.to_csv(f'final_es.csv',index=False)