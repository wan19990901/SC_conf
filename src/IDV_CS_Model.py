import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
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

def trained_LR_model(df_raw,feature_li, report_auroc=False):
    df_concate = prepare_df(df_raw, feature_li)
    split_idx = int(len(df_concate) * 0.8)

    X_train = sm.add_constant(df_concate[feature_li].iloc[:split_idx])
    y_train = df_concate['Correctness'].iloc[:split_idx]
    X_test = sm.add_constant(df_concate[feature_li].iloc[split_idx:])
    y_test = df_concate['Correctness'].iloc[split_idx:]

    # Fit the logistic regression model using statsmodels
    model = sm.Logit(y_train, X_train)
    result = model.fit()
    print(result.summary())

    # Predict probabilities on the entire dataset for consistency
    df_concate['confidence_score'] = result.predict(sm.add_constant(df_concate[feature_li]))
    
    # Create lists of confidence scores
    lists = [df_concate['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_concate), NUM_OF_COT)]
    df_raw['confidence_score'] = lists

    if report_auroc:
        y_pred_proba = result.predict(X_test)
        auroc = roc_auc_score(y_test, y_pred_proba)
        print(f"The AUROC score is: {auroc}")
        return df_raw, auroc

    return df_raw


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