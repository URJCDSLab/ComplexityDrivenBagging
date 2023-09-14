#######################---------- COMPLEXITY-DRIVEN SAMPLING ----------######################################
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from Code.ComplexityMeasures import all_measures
from Code.Hostility_measure_algorithm import hostility_measure
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


root_path = os.getcwd()



def bootstrap_sample(X_train, y_train, weights):
    n_train = len(y_train)
    # Indexes corresponding to a weighted sampling with replacement of the same sample
    # size than the original data
    np.random.seed(1)
    bootstrap_indices = random.choices(np.arange(y_train.shape[0]), weights=weights, k=n_train)

    X_bootstrap = X_train[bootstrap_indices]
    y_bootstrap = y_train[bootstrap_indices]

    return X_bootstrap, y_bootstrap, bootstrap_indices

def voting_rule(preds):

    mode_preds = preds.mode(axis=1)  # most common pred value
    if (mode_preds.shape[1] > 1):
        mode_preds_aux = mode_preds.dropna()  # cases with more than one most common value (= ties)
        np.random.seed(1)
        mode_preds_aux = mode_preds_aux.apply(random.choice, axis=1)  # ties are broken randomly

        mode_preds.iloc[mode_preds_aux.index, 0] = mode_preds_aux

    # Once the ties problem is solved, first column contains the final ensemble predictions
    preds_final = mode_preds[0]

    return preds_final





def ComplexityDrivenBagging(data,n_ensembles, name_data, split,alpha,CM_selected,kfold=10, stump = 'no'):

    """
    :param data: Dataset containing X, y
    :param n_ensembles: ensemble size
    :param name_data: name of the dataset for output results
    :param split: parameter s of Complexity-driven Bagging
    :param alpha: parameter alpha of Complexity-driven Bagging
    :param CM_selected: Complexity measure to guide the sampling
    :param kfold: number of folds for the Stratified k-fold CV, default = 10
    :param stump: default is stump = 'no' to use as base learner unpruned DT
    :return: confusion matrix and accuracy of every learner for every fold
    """

    # Complexity measures list to check
    CM_list = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']

    if (CM_selected not in CM_list):
        raise TypeError("Please select a complexity measure among: Hostility, kDN, DCP, TD_U, CLD, N1, N2, LSC, F1.")

    # X, y
    X = data.iloc[:,:-1].to_numpy() # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy()

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix','accuracy'])


    skf = StratifiedKFold(n_splits=kfold, random_state=1,shuffle=True)
    fold = 0
    for train_index, test_index in skf.split(X, y):
        fold = fold + 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Obtain complexity measures on train set
        data_train = pd.DataFrame(X_train)
        data_train['y'] = y_train
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train,False)

        CM_values = df_measures[CM_selected]

        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        if (alpha >=2):
            quantiles = np.quantile(ranking_hard, q=np.arange(0.5, 0.76, 0.25))
            q50 = quantiles[0]
            q75 = quantiles[1]
            ranking_hard[(ranking_hard >= q75)] = ranking_hard[(ranking_hard >= q75)] * alpha
            ranking_hard[(ranking_hard >= q50) & (ranking_hard < q75)] = ranking_hard[(ranking_hard >= q50) & (
                    ranking_hard < q75)] * (alpha/2)


            quantiles_easy = np.quantile(ranking_easy, q=np.arange(0.5, 0.76, 0.25))
            q50_easy = quantiles_easy[0]
            q75_easy = quantiles_easy[1]
            ranking_easy[(ranking_easy >= q75_easy)] = ranking_easy[(ranking_easy >= q75_easy)] * alpha
            ranking_easy[(ranking_easy >= q50_easy) & (ranking_easy < q75_easy)] = ranking_easy[(
                                                                                                        ranking_easy >= q50_easy) & (
                                                                                                        ranking_easy < q75_easy)] * (alpha/2)
        # if alpha < 2, then no extreme weights are applied


        weights_easy = ranking_easy / sum(ranking_easy)  # probability distribution
        weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
        weights_classic = np.repeat(1 / len(y_train), len(y_train), axis=0)
        w_frac1 = (weights_classic - weights_easy) / split
        w_frac2 = (weights_hard - weights_classic) / split
        weights_v = pd.DataFrame()
        for s in range(split + 1):
            new_w1 = weights_easy + s * w_frac1
            new_w_df1 = pd.DataFrame(new_w1)
            weights_v = pd.concat([weights_v, new_w_df1], axis=1)
        for s in np.arange(1, split + 1):
            new_w2 = weights_classic + s * w_frac2
            new_w_df2 = pd.DataFrame(new_w2)
            weights_v = pd.concat([weights_v, new_w_df2], axis=1)

        preds = pd.DataFrame()
        ensemble_preds = pd.DataFrame()
        j = 0
        for i in range(n_ensembles):

            # Get bootstrap sample following CM_weights
            n_train = len(y_train)

            index_split = weights_v.shape[1] - 1
            if (j <= index_split):
                weights = weights_v.iloc[:, j]
            else:
                j = 0
                weights = weights_v.iloc[:, j]
            j = j + 1

            np.random.seed(1)
            X_bootstrap, y_bootstrap, bootstrap_indices = bootstrap_sample(X_train, y_train, weights)


            # Train DT in bootstrap sample and test y X_test, y_test
            if (stump == 'no'):
                clf = DecisionTreeClassifier(random_state=0)
            else:  # Decision Stump
                clf = DecisionTreeClassifier(max_depth=1, random_state=0)
            clf.fit(X_bootstrap, y_bootstrap)
            y_pred = clf.predict(X_test)

            if (i==0): # first iteration
                col_name = 'pred_' + str(i)
                preds[col_name] = y_pred  # individual predictions
                y_predicted = y_pred
            else:
                col_name = 'pred_'+str(i)
                preds[col_name] = y_pred # individual predictions
                votes = voting_rule(preds)
                votes_dict = {'col_name':votes}
                votes_df = pd.DataFrame(votes_dict)
                votes_df.columns = [col_name]
                ensemble_preds = pd.concat([ensemble_preds, votes_df], axis=1)

                y_predicted = ensemble_preds.iloc[:, -1:] # last column
            acc = accuracy_score(y_predicted, y_test)
            conf_matrix = confusion_matrix(y_test, y_predicted).tolist()



            results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                                'confusion_matrix':[conf_matrix], 'accuracy':acc}
            results_aux = pd.DataFrame(results_dict, index=[0])
            results = pd.concat([results,results_aux])



    return results






