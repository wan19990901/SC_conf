import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utils import *

DATA_DIR = '../data/Evaluation_CoTs/Algo_Design_Data/'

NUM_OF_COT = 40

def customized_LR(coe, intercept, features):
    lincomb = np.dot(coe, features) + intercept
    return 1 / (1 + np.exp(-lincomb))

def customized_LR_model(df_raw, feature_li, coe, intercept, report_auroc=False):
    df_concate = prepare_df(df_raw, feature_li)
    split_idx = int(len(df_concate) * 0.8)
    
    X_test_hard = df_concate[feature_li].iloc[split_idx:]
    y_test_hard = df_concate['Correctness'].iloc[split_idx:]
    
    y_pred_proba = X_test_hard.apply(
        lambda row: customized_LR(coe=coe, intercept=intercept, features=row),
        axis=1
    )

    df_concate['confidence_score'] = df_concate[feature_li].apply(
        lambda row: customized_LR(coe=coe, intercept=intercept, features=row),
        axis=1
    )
    lists = [df_concate['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_concate), NUM_OF_COT)]
    df_raw['confidence_score'] = lists

    if report_auroc:
        auroc = roc_auc_score(y_test_hard, y_pred_proba)
        print("Coefficients:", coe)
        print("Intercept:", intercept)
        print(f"The AUROC score is: {auroc}")
        return df_raw, auroc

    return df_raw

def trained_LR_model(df_raw, feature_li, test_size=0.3, random_state=None, report_auroc=False):
    # Randomly split the raw dataframe into training and testing sets
    df_train_raw, df_test_raw = train_test_split(df_raw, test_size=test_size, random_state=random_state)
        # Reset the index of the training and testing dataframes
    df_train_raw.reset_index(drop=True, inplace=True)
    df_test_raw.reset_index(drop=True, inplace=True)

    # Prepare data for training and testing
    df_train = prepare_df(df_train_raw, feature_li)
    df_test = prepare_df(df_test_raw, feature_li)

    X_train = sm.add_constant(df_train[feature_li])
    y_train = df_train['Correctness']
    X_test = sm.add_constant(df_test[feature_li])
    y_test = df_test['Correctness']

    # Fit the logistic regression model using statsmodels
    model = sm.Logit(y_train, X_train)
    result = model.fit()
    print(result.summary())

    # Predict probabilities on the entire dataset for consistency
    df_train['confidence_score'] = result.predict(X_train)
    df_test['confidence_score'] = result.predict(X_test)
    
    # Aggregate confidence scores in lists for each subset
    df_train_raw['confidence_score'] = [df_train['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_train), NUM_OF_COT)]
    df_test_raw['confidence_score'] = [df_test['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_test), NUM_OF_COT)]

    if report_auroc:
        y_pred_proba = df_test['confidence_score']
        auroc = roc_auc_score(y_test, y_pred_proba)
        print(f"The AUROC score is: {auroc}")
        return df_test_raw, auroc

    return df_test_raw


if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'final_extracted.json')
    df_with_features = pd.read_json(file_path, lines=True)
    feature_li = [
        # 'LEN',
        'QUA_IM',
        'DIF_IV',
        # 'DIF_SUB',
        'SIM_COT_BIGRAM',
        'SIM_COT_AGG',
        # 'SIM_COT_PW',
        'SIM_AC_BIGRAM',
        # 'SIM_AC_AGG',
        'SIM_AC_PW',
    ]
    coe = [-5,-3,-1,1,1,1]
    intercept = -2.5
    # df = trained_LR_model(df_with_flat_features,feature_li, report_auroc=True)
    df = customized_LR_model(df_with_features,feature_li,coe, intercept, report_auroc=True)