import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from pymongo import MongoClient
from clean_data import grab_final_df, make_dummies, trim_df_to_max_teams
import seaborn as sns
import matplotlib.pyplot as plt


def get_base_accuracy(df):
    total_matches = len(df)

    team_odds = df['team_odds'].values
    opp_odds = df['opp_odds'].values
    winner = df['winner'].values

    cut = (1 - abs((1/team_odds) + (1/opp_odds)))/2
    p_team = (1/team_odds) + cut
    p_opp = (1/opp_odds) + cut

    preds = np.array([1 if t > o else 0 for t, o in zip(p_team, p_opp)])

    accuracy = sum(preds == winner)/total_matches

    return accuracy

def get_profit_bet_high_odds(X, y, df):
    total_games = len(X)
    total_profit = 0
    total_bets = 0

    columns = list(df.columns)
    team_odds_ind = columns.index('team_odds')
    opp_odds_ind = columns.index('opp_odds')

    team_odds_list = X[:, team_odds_ind]
    opp_odds_list = X[:, opp_odds_ind]

    for i in range(total_games):
        if team_odds_list[i] < opp_odds_list[i]:
            # Place bet on team >>>
            pred = 1
            total_bets += 1
            if pred == y[i]:
                total_profit += team_odds_list[i]-1
            else:
                total_profit -= 1
        elif opp_odds_list[i] < team_odds_list[i]:
            # Place bet on opponenet >>>
            pred = 0
            total_bets += 1
            if pred == y[i]:
                total_profit += opp_odds_list[i]-1
            else:
                total_profit -= 1

    if total_bets == 0:
        average_profit = 0
    else:
        average_profit = total_profit/total_bets

    # since $1/bet, total_cost = total_bets
    gain = total_profit/total_bets

    print('..........Always Bet on High Odds................')
    print('Total games : {0}\nTotal bets made : {1}\nProfit : ${2:0.2f}\nAverage profit per game : ${3:0.2f}\nGain : {4:0.2f}'.format(total_games, total_bets, total_profit, average_profit, gain))
    return total_profit, total_bets, total_games

def get_profit(model, X_test, y_test, df, threshold):
    total_games = len(X_test)
    total_profit = 0
    total_bets = 0

    columns = list(df.columns)
    team_odds_ind = columns.index('team_odds')
    opp_odds_ind = columns.index('opp_odds')

    team_odds_list = X_test[:, team_odds_ind]
    opp_odds_list = X_test[:, opp_odds_ind]

    probs = model.predict_proba(X_test)
    for i, prob in enumerate(probs):
        if prob[1] > prob[0]:
            pred = 1
            if prob[1] >= threshold:
                # Place bet>>>
                total_bets += 1
                if pred == y_test[i]:
                    team_odds = team_odds_list[i]
                    total_profit += float(team_odds)-1
                else:
                    total_profit -= 1
        if prob[0] > prob[1]:
            pred = 0
            if prob[0] >= threshold:
                # Place bet>>>
                total_bets += 1
                if pred == y_test[i]:
                    opp_odds = opp_odds_list[i]
                    total_profit += float(opp_odds)-1
                else:
                    total_profit -= 1

    if total_bets == 0:
        average_profit = 0
        gain = 0
    else:
        average_profit = total_profit/total_bets
        gain = total_profit/total_bets

    print('..............Threshold Strategy................')
    print('Total games : {0}\nTotal bets made : {1}\nProfit : ${2:0.2f}\nAverage profit per game : ${3:0.2f}\nGain : {4:0.2f}'.format(total_games, total_bets, total_profit, average_profit, gain))
    return total_profit, total_bets, total_games

def get_beat_odds_profit1(model, X_test, y_test, df):
    columns = list(df.columns)
    team_odds_ind = columns.index('team_odds')
    opp_odds_ind = columns.index('opp_odds')

    team_odds_list = X_test[:, team_odds_ind]
    opp_odds_list = X_test[:, opp_odds_ind]

    cut = (1 - abs((1/team_odds_list) + (1/opp_odds_list)))/2
    p_team = (1/team_odds_list) + cut
    p_opp = (1/opp_odds_list) + cut

    model_probs = model.predict_proba(X_test)

    total_games = len(X_test)
    total_profit = 0
    total_bets = 0

    for i, m_probs in enumerate(model_probs):
        m_p_team = m_probs[1]
        m_p_opp = m_probs[0]
        if m_p_team > m_p_opp:
            pred = 1
            if m_p_team > p_team[i]:
                # We beat the odds, place a bet >>>
                total_bets += 1
                if pred == y_test[i]:
                    team_odds = team_odds_list[i]
                    total_profit += team_odds - 1
                else:
                    total_profit -= 1
        else:
            pred = 0
            if m_p_opp > p_opp[i]:
                # We beat the odds, place a bet >>>
                total_bets += 1
                if pred == y_test[i]:
                    opp_odds = opp_odds_list[i]
                    total_profit += opp_odds - 1
                else:
                    total_profit -= 1
    if total_bets == 0:
        average_profit = 0
        gain = 0
    else:
        average_profit = total_profit/total_bets
        gain = total_profit/total_bets

    print('.........Beating Odds Strategy #1..............')
    print('Total games : {0}\nTotal bets made : {1}\nProfit : ${2:0.2f}\nAverage profit per game : ${3:0.2f}\nGain : {4:0.2f}'.format(total_games, total_bets, total_profit, average_profit, gain))
    return total_profit, total_bets, total_games

def get_beat_odds_profit2(model, X_test, y_test, df):
    columns = list(df.columns)
    team_odds_ind = columns.index('team_odds')
    opp_odds_ind = columns.index('opp_odds')

    team_odds_list = X_test[:, team_odds_ind]
    opp_odds_list = X_test[:, opp_odds_ind]

    cut = (1 - abs((1/team_odds_list) + (1/opp_odds_list)))/2
    p_team = (1/team_odds_list) + cut
    p_opp = (1/opp_odds_list) + cut

    model_probs = model.predict_proba(X_test)

    total_games = len(X_test)
    total_profit = 0
    total_bets = 0

    for i, m_probs in enumerate(model_probs):
        m_p_team = m_probs[1]
        m_p_opp = m_probs[0]

        if m_p_team > p_team[i]:
            # We beat market's odds, place bet >>>
            total_bets += 1
            pred = 1
            if pred == y_test[i]:
                team_odds = team_odds_list[i]
                total_profit += team_odds - 1
            else:
                total_profit -= 1
        elif m_p_opp > p_opp[i]:
            # We beat the market's odds, place bet >>>
            total_bets += 1
            pred = 0
            if pred == y_test[i]:
                opp_odds = opp_odds_list[i]
                total_profit += opp_odds - 1
            else:
                total_profit -= 1
    if total_bets == 0:
        average_profit = 0
    else:
        average_profit = total_profit/total_bets

    # since $1/bet and betting each match, total_cost = total_bets
    gain = total_profit/total_bets

    print('.........Beating Odds Strategy #2..............')
    print('Total games : {0}\nTotal bets made : {1}\nProfit : ${2:0.2f}\nAverage profit per game : ${3:0.2f}\nGain : {4:0.2f}'.format(total_games, total_bets, total_profit, average_profit, gain))
    return total_profit, total_bets, total_games

def get_beat_odds_profit3(model, X_test, y_test, df, threshold=0.6, diff=0.1):
    columns = list(df.columns)
    team_odds_ind = columns.index('team_odds')
    opp_odds_ind = columns.index('opp_odds')

    team_odds_list = X_test[:, team_odds_ind]
    opp_odds_list = X_test[:, opp_odds_ind]

    cut = (1 - abs((1/team_odds_list) + (1/opp_odds_list)))/2
    p_team = (1/team_odds_list) + cut
    p_opp = (1/opp_odds_list) + cut

    model_probs = model.predict_proba(X_test)

    total_games = len(X_test)
    total_profit = 0
    total_bets = 0

    for i, m_probs in enumerate(model_probs):
        m_p_team = m_probs[1]
        m_p_opp = m_probs[0]

        if abs(p_team[i] - p_opp[i]) > diff:
            if m_p_team > p_team[i]:
                # We beat market's odds, place bet >>>
                total_bets += 1
                pred = 1
                if pred == y_test[i]:
                    team_odds = team_odds_list[i]
                    total_profit += team_odds - 1
                else:
                    total_profit -= 1
            elif m_p_opp > p_opp[i]:
                # We beat the market's odds, place bet >>>
                total_bets += 1
                pred = 0
                if pred == y_test[i]:
                    opp_odds = opp_odds_list[i]
                    total_profit += opp_odds - 1
                else:
                    total_profit -= 1
        else:
            if m_p_team > m_p_team:
                pred = 1
                if m_p_team >= threshold:
                    # Place bet >>>
                    total_bets += 1
                    if pred == y_test[i]:
                        team_odds = team_odds_list[i]
                        total_profit += team_odds - 1
                    else:
                        total_profit -= 1
            else:
                pred = 0
                if m_p_opp >= threshold:
                    # Place bet >>>
                    total_bets += 1
                    if pred == y_test[i]:
                        opp_odds = opp_odds_list[i]
                        total_profit += opp_odds - 1
                    else:
                        total_profit -= 1


    if total_bets == 0:
        average_profit = 0
        gain = 0
    else:
        average_profit = total_profit/total_bets
        gain = total_profit/total_bets

    print('.........Beating Odds Strategy #3..............')
    print('Total games : {0}\nTotal bets made : {1}\nProfit : ${2:0.2f}\nAverage profit per game : ${3:0.2f}\nGain : {4:0.2f}'.format(total_games, total_bets, total_profit, average_profit, gain))
    return total_profit, total_bets, total_games

def get_beat_odds_profit4(model, X_test, y_test, df):
    ''' In this strategy, the aim is to adjust the amount of money put on a bet according to how much risk the bet is (difference between the market probability and model's probability)'''

    columns = list(df.columns)
    team_odds_ind = columns.index('team_odds')
    opp_odds_ind = columns.index('opp_odds')

    team_odds_list = X_test[:, team_odds_ind]
    opp_odds_list = X_test[:, opp_odds_ind]

    cut = (1 - abs((1/team_odds_list) + (1/opp_odds_list)))/2
    p_team = (1/team_odds_list) + cut
    p_opp = (1/opp_odds_list) + cut

    model_probs = model.predict_proba(X_test)

    total_games = len(X_test)
    total_profit = 0
    total_bets = 0
    max_bet = 4
    total_cost = 0

    for i, m_probs in enumerate(model_probs):
        m_p_team = m_probs[1]
        m_p_opp = m_probs[0]

        if m_p_team > p_team[i]:
            # We beat market's odds, place bet >>>
            total_bets += 1
            pred = 1
            bet_amount = max_bet*(m_p_team - p_team[i])
            # bet_amount = max_bet*m_p_team
            total_cost += bet_amount
            if pred == int(y_test[i]):
                team_odds = team_odds_list[i]
                total_profit += (team_odds*bet_amount) - bet_amount
            else:
                total_profit -= bet_amount
        elif m_p_opp > p_opp[i]:
            # We beat the market's odds, place bet >>>
            total_bets += 1
            pred = 0
            bet_amount = max_bet*(m_p_opp - p_opp[i])
            # bet_amount = max_bet*m_p_opp
            total_cost += bet_amount
            if pred == int(y_test[i]):
                opp_odds = opp_odds_list[i]
                total_profit += (opp_odds*bet_amount) - bet_amount
            else:
                total_profit -= bet_amount
    if total_bets == 0:
        average_profit = 0
    else:
        average_profit = total_profit/total_bets

    gain = total_profit/total_cost

    print('.........Beating Odds Strategy #4..............')
    print('Total games : {0}\nTotal bets made : {1}\nTotal Cost : ${2:0.2f}\nProfit : ${3:0.2f}\nAverage profit per game : ${4:0.2f}\nGain : {5:0.2f}'.format(total_games, total_bets, total_cost, total_profit, average_profit, gain))
    return total_profit, total_bets, total_games

def class_crossval(data, models):
    seed = 123
    names = []
    rmse = []
    mae = []
    scoring = 'accuracy'
    for name, model in models:
        scores = evaluate(model, data, measures=['RMSE', 'MAE'])
        rmse_single = scores['rmse']
        mae_single = scores['mae']
        rmse.append(rmse_single)
        mae.append(mae_single)
        names.append(name)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison of CrossVal Scores')
    ax1 = fig.add_subplot(121)
    # ax1.set_title('RSME')
    ax1.set_ylabel('RSME')
    sns.boxplot(data=rmse, orient='v', ax=ax1)
    # plt.boxplot(rmse)
    ax1.set_xticklabels(names)
    ax2 = fig.add_subplot(122)
    # ax2.set_title('MAE')
    ax2.set_ylabel('MAE')
    sns.boxplot(data=mae, orient='v', ax=ax2)
    # plt.boxplot(mae)
    ax2.set_xticklabels(names)
    plt.show()

def plot_cv_scores(df, models):
    names, accuracies = [], []

    # Dropping all time data information:
    df.drop(['date', 'day_of_month', 'day_of_week', 'month', 'year', 'hour', 'min'], axis=1, inplace=True)
    df2 = df.copy()
    y = df.pop('winner').values
    X = df.values
    cat_features = [0, 1, 2, 9]
    df2 = make_dummies(df2, 'team')
    df2 = make_dummies(df2, 'opponent')
    df2 = make_dummies(df2, 'match_type')
    df2 = make_dummies(df2, 'rank_version')
    y2 = df2.pop('winner').values
    X2 = df2.values

    tscv = TimeSeriesSplit(n_splits=3)
    cat_scores = []

    for name, model in models:
        scores = []
        for train_ind, test_ind in tscv.split(X):
            if name == 'Cat Boost':
                model.fit(X[train_ind], y[train_ind], cat_features)
                score = model.score(X[test_ind], y[test_ind])
            else:
                model.fit(X2[train_ind], y2[train_ind])
                score = model.score(X2[test_ind], y2[test_ind])
            scores.append(score)
        accuracies.append(scores)
        names.append(name)
    fig = plt.figure()
    fig.suptitle('Model Comparison: CrossVal Scores (Non-randomized)')
    ax = fig.add_subplot(111)
    ax.set_ylabel('Accuracy')
    sns.boxplot(data=accuracies, orient='v', ax=ax)
    ax.set_xticklabels(names)
    ax.figure.subplots_adjust(bottom = 0.2)
    plt.xticks(rotation=45)
    plt.show()
    return names, accuracies

def plot_models(df, models):
    names, accuracies = [], []

    # Dropping all time data information:
    df.drop(['date', 'day_of_month', 'day_of_week', 'month', 'year', 'hour', 'min'], axis=1, inplace=True)
    df2 = df.copy()
    y = df.pop('winner').values
    X = df.values
    cat_features = [0, 1, 2, 9]
    df2 = make_dummies(df2, 'team')
    df2 = make_dummies(df2, 'opponent')
    df2 = make_dummies(df2, 'match_type')
    df2 = make_dummies(df2, 'rank_version')
    y2 = df2.pop('winner').values
    X2 = df2.values

    tscv = TimeSeriesSplit(n_splits=3)
    cat_scores = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, test_size=0.33, shuffle=False)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, stratify=None, test_size=0.33, shuffle=False)

    for name, model in models:
        scores = []
        if name == 'Cat Boost':
            model.fit(X_train, y_train, cat_features)
            score = model.score(X_test, y_test)
        else:
            model.fit(X_train2, y_train2)
            score = model.score(X_test2, y_test2)
        accuracies.append(score)
        names.append(name)
    fig = plt.figure()
    fig.suptitle('Model Comparison: Test Accuracy (No Shuffling)')
    ax = fig.add_subplot(111)
    ax.set_ylabel('Accuracy')
    sns.barplot(x=names, y=accuracies, orient='v', ax=ax)
    ax.set_xticklabels(names)
    ax.figure.subplots_adjust(bottom = 0.2)
    plt.xticks(rotation=45)
    plt.show()

    return names, accuracies

class BinnedModel():
    def __init__(self, bin_num=4):
        self.bin_num = bin_num

    def fit(self, X, y, cat_features):
        odds_diff = abs(X[:, 8] - X[:, 9])
        binned_vals = pd.qcut(odds_diff, self.bin_num)
        self.bins = binned_vals.categories
        self.models = {}
        y = np.array(y)
        for i, b in enumerate(self.bins):
            X_bin = np.copy(X[binned_vals == b])
            y_bin = np.copy(y[binned_vals == b])
            cat = CatBoostClassifier(logging_level='Silent')
            print('Fitting model : {}'.format(i+1))
            cat.fit(X_bin, y_bin, cat_features)
            self.models[b] = cat

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for i, row in enumerate(X):
            odds_diff = abs(X[i, 8] - X[i, 9])
            prob = []
            for b in self.bins:
                if odds_diff in b:
                    prob = self.models[b].predict_proba(np.array([list(row)]))[0]
                    probs[i] = prob
            if prob == []:
                prob = self.models[self.bins[-1]].predict_proba(np.array([list(row)]))[0]
                probs[i] = prob

        return probs

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for i, row in enumerate(X):
            odds_diff = abs(X[i, 8] - X[i, 9])
            prob = []
            for b in self.bins:
                if odds_diff in b:
                    prob = self.models[b].predict_proba(np.array([list(row)]))[0]
                    probs[i] = prob
            if prob == []:
                prob = self.models[self.bins[-1]].predict_proba(np.array([list(row)]))[0]
                probs[i] = prob

        return probs

def get_best_grid_searched_model(p_grid, clf, X_train, X_test, y_train, y_test, df):

    grid_model = GridSearchCV(estimator=clf, param_grid=p_grid, n_jobs=-1, verbose=1)

    grid_model.fit(X_train, y_train)

    best_clf = grid_model.best_estimator_
    print('...............................')
    print('Best model parameters : \n{}'.format(grid_model.best_params_))
    print('Best model accuracy : {}'.format(best_clf.score(X_test, y_test)))

    profit_all_th, bets_all_th, games_all_th = get_profit(best_clf, X_test, y_test, df, threshold=0.6)

    profit_all_bo1, bets_all_bo1, games_all_bo1 = get_beat_odds_profit1(best_clf, X_test, y_test, df)

    profit_all_bo2, bets_all_bo2, games_all_bo2 = get_beat_odds_profit2(best_clf, X_test, y_test, df)

    profit_all_bo3, bets_all_bo3, games_all_bo3 = get_beat_odds_profit3(best_clf, X_test, y_test, df, threshold=0.6, diff=0.1)

    profit_all_bo4, bets_all_bo4, games_all_bo4 = get_beat_odds_profit4(best_clf, X_test, y_test, df)

    return grid_model

if __name__ == '__main__':

    df_full = grab_final_df('team_stats_90d_ago', max_games=1)

    df = df_full.copy()

    # Cross Valdiation Tests for different Classification Models:
    models = []
    models.append(('Random Forest', RandomForestClassifier()))
    models.append(('Ada Boost', AdaBoostClassifier()))
    models.append(('Gradient Boost', GradientBoostingClassifier()))
    models.append(('MLP', MLPClassifier()))
    models.append(('Cat Boost', CatBoostClassifier(logging_level='Silent', od_type='IncToDec', od_pval=0.5, depth=6)))
    # model_names, accuracies = plot_models(df, models)
    model_names, accuracies = plot_cv_scores(df, models)

    # Dropping all time data information:
    df.drop(['date', 'day_of_month', 'day_of_week', 'month', 'year', 'hour', 'min'], axis=1, inplace=True)
    y = df.pop('winner').values
    X = df.values
    cat_features = [0, 1, 2, 9]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, test_size=0.33, shuffle=False)

    df_train, df_test = train_test_split(df, shuffle=False)

    cat = CatBoostClassifier(logging_level='Silent', od_type='IncToDec', od_pval=0.5, depth=6)
    cat.fit(X_train, y_train, cat_features, use_best_model=True, eval_set=(X_test, y_test))
    train_accuracy = cat.score(X_train, y_train)
    test_accuracy = cat.score(X_test, y_test)
    print('Train accuracy without Odds features : {}'.format(train_accuracy))
    print('Test accuracy without Odds features: {}'.format(test_accuracy))

    base_accuracy = get_base_accuracy(df_full)
    print('Base Accuracy : {}'.format(base_accuracy))

    profit_all_ho, bets_all_ho, games_all_ho = get_profit_bet_high_odds(X, y, df)

    profit_all_th, bets_all_th, games_all_th = get_profit(cat, X_test, y_test, df, threshold=0.6)

    profit_all_bo1, bets_all_bo1, games_all_bo1 = get_beat_odds_profit1(cat, X_test, y_test, df)

    profit_all_bo2, bets_all_bo2, games_all_bo2 = get_beat_odds_profit2(cat, X_test, y_test, df)

    profit_all_bo3, bets_all_bo3, games_all_bo3 = get_beat_odds_profit3(cat, X_test, y_test, df, threshold=0.6, diff=0.1)

    profit_all_bo4, bets_all_bo4, games_all_bo4 = get_beat_odds_profit4(cat, X_test, y_test, df)


    tscv = TimeSeriesSplit(n_splits=3)
    cat_scores = []
    for train_ind, test_ind in tscv.split(X):
        cat = CatBoostClassifier(logging_level='Silent')
        cat_features = [0, 1, 2, 9]
        cat.fit(X[train_ind], y[train_ind], cat_features=cat_features)
        s = cat.score(X[test_ind], y[test_ind])
        print('accuracy : {}'.format(s))
        cat_scores.append(s)


    ### Testing SKlearn's Decision Tree Ensembles #############
    '''
    oss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’
    '''
    df2 = df_full.copy()
    df2.drop(['date', 'day_of_month', 'day_of_week', 'month', 'year', 'hour', 'min'], axis=1, inplace=True)
    df2 = make_dummies(df2, 'team')
    df2 = make_dummies(df2, 'opponent')
    df2 = make_dummies(df2, 'match_type')
    df2 = make_dummies(df2, 'rank_version')
    y2 = df2.pop('winner').values
    X2 = df2.values

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, stratify=None, test_size=0.33, shuffle=False)

    p_grid_ada = {'n_estimators' : [50, 100, 300],
                'learning_rate' : [0.1, 1.0],
                'algorithm': ['SAMME.R', 'SAMME']}

    grid_ada = get_best_grid_searched_model(p_grid_ada, AdaBoostClassifier(), X_train2, X_test2, y_train2, y_test2, df2)

    p_grid_rf = {'n_estimators' : [50, 100, 300],
                'criterion':['gini'],
                'max_depth' : [None],
                'min_samples_split' : [2, 3],
                'min_samples_leaf' : [1],
                'min_weight_fraction_leaf':[0.0],
                'max_features' : ['auto'],
                'max_leaf_nodes' : [None],
                'min_impurity_decrease' : [0.0],
                'min_impurity_split' : [None],
                'bootstrap' : [True],
                'oob_score' : [False],
                'n_jobs' : [-1]}

    grid_rf = get_best_grid_searched_model(p_grid_rf, RandomForestClassifier(), X_train2, X_test2, y_train2, y_test2, df2)

    p_grid_gcb = {'learning_rate': [0.1, 0.2],
                'n_estimators' : [100, 130, 150],
                'max_depth' : [1, 2]}

    grid_gbc = get_best_grid_searched_model(p_grid_gcb, GradientBoostingClassifier(), X_train2, X_test2, y_train2, y_test2, df2)

    p_grid_mlp = {'hidden_layer_sizes' : [(100, 100), (200,)],
                'activation' : ['identity','relu', 'tanh', 'logistic'],
                'max_iter' : [200, 300]}

    grid_mlp = get_best_grid_searched_model(p_grid_mlp, MLPClassifier(), X_train2, X_test2, y_train2, y_test2, df2)
