import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from pymongo import MongoClient
from clean_data import grab_final_df


def get_base_accuracy(df):
    total_matches = len(df)

    team_odds = df['team_odds']
    opp_odds = df['opp_odds']
    winner = df['winner']

    preds = np.array([1 if t < o else 0 for t, o in zip(team_odds, opp_odds)])

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

if __name__ == '__main__':

    df_full = grab_final_df('team_stats_90d_ago', max_games=1)

    df = df_full.copy()

    # Dropping all time data information:
    df.drop(['date', 'day_of_month', 'day_of_week', 'month', 'year', 'hour', 'min'], axis=1, inplace=True)
    y = df.pop('winner').values
    X = df.values
    cat_features = [0, 1, 2, 9]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, shuffle=True)

    cat = CatBoostClassifier(logging_level='Verbose', od_type='IncToDec', od_pval=0.5, depth=6)
    cat.fit(X_train, y_train, cat_features, use_best_model=True, eval_set=(X_test, y_test))
    train_accuracy = cat.score(X_train, y_train)
    test_accuracy = cat.score(X_test, y_test)
    print('Train accuracy : {}'.format(train_accuracy))
    print('Test accuracy : {}'.format(test_accuracy))

    base_accuracy = get_base_accuracy(df_full)
    print('Base Accuracy : {}'.format(base_accuracy))

    profit_all_ho, bets_all_ho, games_all_ho = get_profit_bet_high_odds(X, y, df)

    profit_all_th, bets_all_th, games_all_th = get_profit(cat, X_test, y_test, df, threshold=0.6)

    profit_all_bo1, bets_all_bo1, games_all_bo1 = get_beat_odds_profit1(cat, X_test, y_test, df)

    profit_all_bo2, bets_all_bo2, games_all_bo2 = get_beat_odds_profit2(cat, X_test, y_test, df)

    profit_all_bo3, bets_all_bo3, games_all_bo3 = get_beat_odds_profit3(cat, X_test, y_test, df, threshold=0.6, diff=0.1)

    profit_all_bo4, bets_all_bo4, games_all_bo4 = get_beat_odds_profit4(cat, X_test, y_test, df)

    ########### Grid Searching for optimized parameters ##############
    # '''
    #                     iterations=500,
    #                     learning_rate=0.03,
    #                     depth=6,
    #                     l2_leaf_reg=3,
    #                      rsm=None,
    #                      loss_function='Logloss',
    #                      border_count=None,
    #                      feature_border_type='MinEntropy',
    #                      fold_permutation_block_size=None,
    #                      od_pval=None,
    #                      od_wait=None,
    #                      od_type=None,
    #                      nan_mode=None,
    #                      counter_calc_method=None,
    #                      leaf_estimation_iterations=None,
    #                      leaf_estimation_method=None,
    #                      thread_count=None,
    #                      random_seed=None,
    #                      use_best_model=None,
    #                      verbose=None,
    #                      logging_level=None,
    #                      ctr_description=None,
    #                      ctr_border_count=None,
    #                      ctr_leaf_count_limit=None,
    #                      store_all_simple_ctr=None,
    #                      max_ctr_complexity=None,
    #                      priors=None,
    #                      has_time=False,
    #                      classes_count=None,
    #                      class_weights=None,
    #                      one_hot_max_size=None,
    #                      random_strength=1,
    #                      name='experiment',
    #                      ignored_features=None,
    #                      train_dir=None,
    #                      custom_loss=None,
    #                      custom_metric=None,
    #                      eval_metric=None,
    #                      bagging_temperature=None,
    #                      save_snapshot=None,
    #                      snapshot_file=None,
    #                      fold_len_multiplier=None,
    #                      used_ram_limit=None,
    #                      feature_priors=None,
    #                      allow_writing_files=None,
    #                      approx_on_full_history=None,
    #                      task_type=None,
    #                      device_config=None
    # '''
    #
    # p_grid = {'learning_rate': [0.1],
    #             'depth' : [2, 6, 8],
    #             'loss_function' : ['Logloss', 'CrossEntropy']}
    #
    # cat_grid_model = GridSearchCV(estimator=CatBoostClassifier, param_grid=p_grid, n_jobs=-1, verbose=1)
    #
    # cat_grid_model.fit(X_train, y_train.reshape((y_train.shape[0], 1)))
    #
    # best_cat = cat_grid_model.best_estimator_
    # print('...............................')
    # print('Best model parameters : \n{}'.format(cat_grid_model.best_params_))
    # print('Best model score : {}'.format(best_cat.score(X_test, y_test)))
