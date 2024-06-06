import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from utils import *

DATA_DIR = '../data/Evaluation_CoTs/Algo_Design_Data/'

NUM_OF_COT = 40

def customized_LR(coe,  features,intercept):
    lincomb = np.dot(coe, features) + intercept
    return 1 / (1 + np.exp(-lincomb))

def customized_LR_model(df_raw, feature_li, coe, intercept,report_auroc=False):
    # Prepare data for testing
    df_test = prepare_df(df_raw, feature_li)

    X_test = sm.add_constant(df_test[feature_li])
    y_test = df_test['Correctness']
    y_pred_proba = X_test.apply(
        lambda row: customized_LR(coe=coe, features=row[feature_li],intercept = intercept),
        axis=1
    )

    X_test['confidence_score'] = y_pred_proba
    df_raw['confidence_score'] = [X_test['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_test), NUM_OF_COT)]


    if report_auroc:
        auroc = roc_auc_score(y_test, y_pred_proba)
        print(f"The AUROC score is: {auroc}")
        return df_raw, auroc

    return df_raw

def train_test_split_stratify(df,test_size,random_state):

    df['Stratify'] = df['Model'] + '_' + df['Name']

    # Initialize StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Split the data
    for train_index, test_index in split.split(df, df['Stratify']):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

    # Drop the 'Stratify' column if it's no longer needed
    train_df = train_df.drop(columns=['Stratify'])
    test_df = test_df.drop(columns=['Stratify'])
    # Display the result
    print("Train DataFrame:")
    print(train_df['Model'].value_counts())
    print(train_df['Name'].value_counts())

    print("\nTest DataFrame:")
    print(test_df['Model'].value_counts())
    print(test_df['Name'].value_counts())
    return train_df,test_df


def trained_LR_model(df_raw, feature_li, test_size=0.3, random_state=2024, report_auroc=False, train_mode = False):
    # Randomly split the raw dataframe into training and testing sets
    # df_train_raw, df_test_raw = train_test_split(df_raw, test_size=test_size, random_state=random_state)
    df_train_raw, df_test_raw = train_test_split_stratify(df_raw, test_size=test_size, random_state=random_state)
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
    if train_mode:
        coef = result.params.values
        return df_test_raw, coef

    return df_test_raw

def trained_RF_model(df_raw, feature_li, test_size=0.3, random_state=2024, report_auroc=False, train_mode=False):
    # Randomly split the raw dataframe into training and testing sets
    df_train_raw, df_test_raw = train_test_split_stratify(df_raw, test_size=test_size, random_state=random_state)
    
    # Reset the index of the training and testing dataframes
    df_train_raw.reset_index(drop=True, inplace=True)
    df_test_raw.reset_index(drop=True, inplace=True)

    # Prepare data for training and testing
    df_train = prepare_df(df_train_raw, feature_li)
    df_test = prepare_df(df_test_raw, feature_li)
    X_train = df_train[feature_li]
    y_train = df_train['Correctness']
    X_test = df_test[feature_li]
    y_test = df_test['Correctness']

    # Train the Random Forest classifier
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)

    # Predict probabilities on the entire dataset for consistency
    df_train['confidence_score'] = rf.predict_proba(X_train)[:, 1]
    df_test['confidence_score'] = rf.predict_proba(X_test)[:, 1]

    # Aggregate confidence scores in lists for each subset
    df_train_raw['confidence_score'] = [df_train['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_train), NUM_OF_COT)]
    df_test_raw['confidence_score'] = [df_test['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_test), NUM_OF_COT)]

    if report_auroc:
        y_pred_proba = df_test['confidence_score']
        auroc = roc_auc_score(y_test, y_pred_proba)
        print(f"The AUROC score is: {auroc}")
        return df_test_raw, auroc
    
    if train_mode:
        feature_importances = rf.feature_importances_
        return df_test_raw, feature_importances

    return df_test_raw

def trained_NB_model(df_raw, feature_li, test_size=0.3, random_state=2024, report_auroc=False, train_mode=False):
    # Randomly split the raw dataframe into training and testing sets
    df_train_raw, df_test_raw = train_test_split_stratify(df_raw, test_size=test_size, random_state=random_state)

    # Reset the index of the training and testing dataframes
    df_train_raw.reset_index(drop=True, inplace=True)
    df_test_raw.reset_index(drop=True, inplace=True)

    # Prepare data for training and testing
    df_train = prepare_df(df_train_raw, feature_li)
    df_test = prepare_df(df_test_raw, feature_li)

    X_train = df_train[feature_li]
    y_train = df_train['Correctness']
    X_test = df_test[feature_li]
    y_test = df_test['Correctness']

    # Train the Naive Bayes classifier
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Predict probabilities on the entire dataset for consistency
    df_train['confidence_score'] = nb.predict_proba(X_train)[:, 1]
    df_test['confidence_score'] = nb.predict_proba(X_test)[:, 1]

    # Aggregate confidence scores in lists for each subset
    df_train_raw['confidence_score'] = [df_train['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_train), NUM_OF_COT)]
    df_test_raw['confidence_score'] = [df_test['confidence_score'].iloc[i:i + NUM_OF_COT].tolist() for i in range(0, len(df_test), NUM_OF_COT)]

    if report_auroc:
        y_pred_proba = df_test['confidence_score']
        auroc = roc_auc_score(y_test, y_pred_proba)
        print(f"The AUROC score is: {auroc}")
        return df_test_raw, auroc

    if train_mode:
        # No support vectors in Naive Bayes, return None instead
        return df_test_raw, None

    return df_test_raw


if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'final_extracted.json')
    df_with_features = pd.read_json(file_path, lines=True)
    feature_li =['LEN','QUA_IM','DIF_IV','SIM_INPUT','SIM_COT_BIGRAM']
    coe = [-5,-3,-1,1,1,1]
    intercept = -2.5
    df = trained_LR_model(df_with_features,feature_li, report_auroc=True,train_mode=True)
    # df = customized_LR_model(df_with_features,feature_li,coe, intercept, report_auroc=True)