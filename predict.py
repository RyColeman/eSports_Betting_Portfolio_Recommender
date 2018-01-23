import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import scrape_team_stats_csgo as scrape
from catboost import CatBoostClassifier

def scrape_rankings(date, ranking_coll):
    ua = UserAgent()
    driver = scrape.init_driver(ua)
    url = 'https://www.hltv.org/stats/teams?minMapCount=0'
    driver.get(url)
    scrape.wait()
    yesterday = date - pd.Timedelta('1 day')
    scrape.change_url_date(driver, yesterday, 90)
    rankings_xpath = '/html/body/div[2]/div/div[2]/div[1]/div/table'
    try:
        rankings = driver.find_element_by_xpath(rankings_xpath).text.split('\n')
    except:
        scrape.wait()
        rankings = driver.find_element_by_xpath(rankings_xpath).text.split('\n')

    ranking_coll.update_one({'date' : date}, {'$set' : {'rankings' : rankings}}, upsert=True)

    driver.quit()

    return rankings

def get_ranking_info(date, team, opponent, ranking_coll):
    # See if there's ranking data for this date, if not, scrape and add to MongoDB:
    r = []
    c = ranking_coll.find({'date' : date})
    for doc in c:
        r.append(doc)

    if r == []:
        rankings = scrape_rankings(date, ranking_coll)
    else:
        rankings = r[0]['rankings']

    rank_version = float(rankings[1])
    team_has_ratings = False
    opp_has_ratings = False
    for line in rankings[2:]:
        if team in line.lower().replace(' ', ''):
            team_has_ratings = True
            team_rank = line.split(' ')[-4:]
            maps_played_team = int(team_rank[0])
            KD_diff_team = int(team_rank[1])
            KD_team = float(team_rank[2])
            ranking_team = float(team_rank[3])
        if opponent in line.lower().replace(' ', ''):
            opp_has_ratings = True
            opp_rank = line.split(' ')[-4:]
            maps_played_opp = int(opp_rank[0])
            KD_diff_opp = int(opp_rank[1])
            KD_opp = float(opp_rank[2])
            ranking_opp = float(opp_rank[3])

    if team_has_ratings == False:
        maps_played_team = np.nan
        KD_diff_team = np.nan
        KD_team = np.nan
        ranking_team = np.nan

    if opp_has_ratings == False:
        maps_played_opp = np.nan
        KD_diff_opp = np.nan
        KD_opp = np.nan
        ranking_opp = np.nan

    return (rank_version, maps_played_team, KD_diff_team, KD_team, ranking_team, maps_played_opp, KD_diff_opp, KD_opp, ranking_opp, team_has_ratings, opp_has_ratings)

def get_predictions(model, df, date, time, team, opponent, team_odds, opp_odds, ranking_coll, match_type='', print_out=True):
    ''' order of features for model_v2
    ['match_type',
    'opponent',
    'team',
    'team_ss',
    'opp_ss',
    'team_win_ratio',
    'opp_win_ratio',
    'opp_odds',
    'team_odds',
    'rank_version',
    'maps_played_team',
    'maps_played_opp',
    'KD_diff_team',
    'KD_diff_opp',
    'KD_team',
    'KD_opp',
    'ranking_team',
    'ranking_opp']
    '''

    ''' date format: 'YYYY-MM-DD' (here, MM and DD can be 1 or 2 numbers in length)
    time: 'HH:MM' (HH can be 1 or 2 numbers in length)
    '''
    # Making sure teams are correctly formatted:
    team = team.lower().replace(' ', '')
    opponent = opponent.lower().replace(' ', '')

    h, m = time.split(':')
    time_delta = pd.Timedelta('{0}h, {1}m'.format(h, m))
    time_zone_change = pd.Timedelta('1h')

    # Back to MTN where model is trained on:
    date_time = pd.to_datetime(date) + time_delta + time_zone_change
    date = pd.to_datetime('{0}-{1}-{2}'.format(date_time.year, date_time.month, date_time.day))
    day_of_month = date_time.day
    day_of_week = date_time.dayofweek
    month = date_time.month
    hour = date_time.hour
    minute = date_time.minute

    # Putting odds information into MongoDB:
    # if odds_coll != None:
    #     odds_coll.update_one({'date' : date_time, 'map' : match_type, 'team1' : team, 'team2' : opponent}, {'$set' : {'odds1' : team_odds, 'odds2' : opp_odds}}, upsert=True)


    # geting all the unique teams present when training the model:
    all_teams = set(df['team'].unique())

    (rank_version, maps_played_team, KD_diff_team, KD_team, ranking_team, maps_played_opp, KD_diff_opp, KD_opp, ranking_opp, team_has_ratings, opp_has_ratings) = get_ranking_info(date, team, opponent, ranking_coll)

    team_in_all_teams = False
    opp_in_all_teams = False

    if team in all_teams:
        team_in_all_teams = True
    if opponent in all_teams:
        opp_in_all_teams = True

    should_I_bet = True
    reason = "There is rating data for each team and each team was present in the model's training data"
    warnings = []
    if team_in_all_teams == False:
        should_I_bet = False
        warning1 = "Warning: team : {} was not present in model's training data".format(team)
        warnings.append(warning1)
    if opp_in_all_teams == False:
        should_I_bet = False
        warning2 = "Warning: opponent : {} was not present in model's training data".format(opponent)
        warnings.append(warning2)
    if team_has_ratings == False:
        should_I_bet = False
        warning3 = "Warning: team : {} did not have ratings".format(team)
        warnings.append(warning3)
    if opp_has_ratings == False:
        should_I_bet = False
        warning4 = "Warning: oppenent : {} did not have ratings".format(opponent)
        warnings.append(warning4)

    X_new = np.array([[match_type, opponent, team, team_ss, opp_ss, team_win_ratio, opp_win_ratio, opp_odds, team_odds, rank_version, maps_played_team, maps_played_opp, KD_diff_team, KD_diff_opp, KD_team, KD_opp, ranking_team, ranking_opp]])

    probs = model.predict_proba(X_new)[0]

    if print_out:
        if warnings != []:
            for warning in warnings:
                print(warning)

    if should_I_bet:
        advice = [should_I_bet, reason]
    else:
        advice = [should_I_bet, warnings]


    return probs, advice

def get_predictions_v2(model, df, date, time, team, opponent, team_odds, opp_odds, ranking_coll, match_type, print_out=True):
    ''' order of features without time:
    ['match_type',
    'opponent',
    'team',
    'opp_odds',
    'team_odds',
    'rank_version',
    'maps_played_team',
    'maps_played_opp',
    'KD_diff_team',
    'KD_diff_opp',
    'KD_team',
    'KD_opp',
    'ranking_team',
    'ranking_opp']

    date format: 'YYYY-MM-DD' (here, MM and DD can be 1 or 2 numbers in length)
    time: 'HH:MM' (HH can be 1 or 2 numbers in length)
    '''
    # Making sure teams are correctly formatted:
    team = team.lower().replace(' ', '')
    opponent = opponent.lower().replace(' ', '')

    h, m = time.split(':')
    time_delta = pd.Timedelta('{0}h, {1}m'.format(h, m))
    time_zone_change = pd.Timedelta('1h')

    # Back to MTN where model is trained on:
    date_time = pd.to_datetime(date) + time_delta + time_zone_change
    date = pd.to_datetime('{0}-{1}-{2}'.format(date_time.year, date_time.month, date_time.day))
    day_of_month = date_time.day
    day_of_week = date_time.dayofweek
    month = date_time.month
    year = date_time.year

    # geting all the unique teams present when training the model:
    all_teams = set(df['team'].unique())

    (rank_version, maps_played_team, KD_diff_team, KD_team, ranking_team, maps_played_opp, KD_diff_opp, KD_opp, ranking_opp, team_has_ratings, opp_has_ratings) = get_ranking_info(date, team, opponent, ranking_coll)

    team_in_all_teams = False
    opp_in_all_teams = False

    if team in all_teams:
        team_in_all_teams = True
    if opponent in all_teams:
        opp_in_all_teams = True

    should_I_bet = True
    reason = "There is rating data for each team and each team was present in the model's training data"
    warnings = []
    if team_in_all_teams == False:
        should_I_bet = False
        warning1 = "Warning: team : {} was not present in model's training data".format(team)
        warnings.append(warning1)
    if opp_in_all_teams == False:
        should_I_bet = False
        warning2 = "Warning: opponent : {} was not present in model's training data".format(opponent)
        warnings.append(warning2)
    if team_has_ratings == False:
        should_I_bet = False
        warning3 = "Warning: team : {} did not have ratings".format(team)
        warnings.append(warning3)
    if opp_has_ratings == False:
        should_I_bet = False
        warning4 = "Warning: oppenent : {} did not have ratings".format(opponent)
        warnings.append(warning4)

    # For Model v2
    X_new = np.array([[match_type, opponent, team, opp_odds, team_odds, rank_version, maps_played_team, maps_played_opp, KD_diff_team, KD_diff_opp, KD_team, KD_opp, ranking_team, ranking_opp]])

    probs = model.predict_proba(X_new)[0]

    if print_out:
        if warnings != []:
            for warning in warnings:
                print(warning)

    if should_I_bet:
        advice = [should_I_bet, reason]
    else:
        advice = [should_I_bet, warnings]


    return probs, advice

def calc_profit_model_probs(model, df, X, ranking_coll, threshold=.6):
    total_profit = 0
    total_cost = 0

    for i, row in enumerate(X):
        date = row[0]
        time = row[1]
        team = row[2]
        opponent = row[3]
        team_odds = float(row[4])
        opp_odds = float(row[5])
        match_type = row[6]
        winner = row[7]

        probs, advice = get_predictions(model=model, df=df, date=date, time=time, team=team, opponent=opponent, team_odds=team_odds, opp_odds=opp_odds, ranking_coll=ranking_coll, match_type=match_type, print_out=False)

        should_I_bet = advice[0]
        if should_I_bet:

            if probs[1] > probs[0] and probs[1] > threshold:
                pred = '1'
                total_cost += 1
                if pred == winner:
                    # winn bet
                    total_profit += team_odds - 1
                else:
                    total_profit -= 1
            elif probs[0] > probs[1] and probs[0] > threshold:
                pred = '0'
                total_cost += 1
                if pred == winner:
                    # winn bet
                    total_profit += opp_odds - 1
                else:
                    total_profit -= 1

    if total_cost == 0:
        gain = 0
    else:
        gain = total_profit/total_cost

    return total_profit, total_cost, gain

def calc_profit(model, df, X, ranking_coll):
    # X = [dates, times, teams, opponents, team_odds, opp_odds, match_types, winner]

    bet = 10
    total_profit = 0
    total_cost = 0

    for i, row in enumerate(X):
        date = row[0]
        time = row[1]
        team = row[2]
        opponent = row[3]
        team_odds = float(row[4])
        opp_odds = float(row[5])
        match_type = row[6]
        winner = row[7]

        probs, advice = get_predictions(model=model, df=df, date=date, time=time, team=team, opponent=opponent, team_odds=team_odds, opp_odds=opp_odds, ranking_coll=ranking_coll, match_type=match_type, print_out=False)

        should_I_bet = advice[0]
        if should_I_bet:

            bet_multiplier, bet_team = get_recommendation(probs, team, opponent, team_odds, opp_odds, print_out=False)

            if bet_team == team:
                pred = '1'
                bet_amount = bet*bet_multiplier
                total_cost += bet_amount
                if pred == winner:
                    # winn bet
                    total_profit += team_odds*bet_amount - bet_amount
                else:
                    total_profit -= bet_amount
            else:
                pred = '0'
                bet_amount = bet*bet_multiplier
                total_cost += bet_amount
                if pred == winner:
                    # winn bet
                    total_profit += opp_odds*bet_amount - bet_amount
                else:
                    total_profit -= bet_amount

    if total_cost == 0:
        gain = 0
    else:
        gain = total_profit/total_cost

    return total_profit, total_cost, gain

def calc_profit_v2(model, df, X, ranking_coll):
    # X = [dates, times, teams, opponents, team_odds, opp_odds, match_types, winner]

    bet = 10
    total_profit = 0
    total_cost = 0

    for i, row in enumerate(X):
        date = row[0]
        time = row[1]
        team = row[2]
        opponent = row[3]
        team_odds = float(row[4])
        opp_odds = float(row[5])
        match_type = row[6]
        winner = row[7]

        probs, advice = get_predictions_v2(model=model, df=df, date=date, time=time, team=team, opponent=opponent, team_odds=team_odds, opp_odds=opp_odds, ranking_coll=ranking_coll, match_type=match_type, print_out=False)

        should_I_bet = advice[0]
        if should_I_bet:

            bet_multiplier, bet_team = get_recommendation(probs, team, opponent, team_odds, opp_odds, print_out=False)

            if bet_team == team:
                pred = '1'
                bet_amount = bet*bet_multiplier
                total_cost += bet_amount
                if pred == winner:
                    # winn bet
                    total_profit += team_odds*bet_amount - bet_amount
                else:
                    total_profit -= bet_amount
            else:
                pred = '0'
                bet_amount = bet*bet_multiplier
                total_cost += bet_amount
                if pred == winner:
                    # winn bet
                    total_profit += opp_odds*bet_amount - bet_amount
                else:
                    total_profit -= bet_amount

    if total_cost == 0:
        gain = 0
    else:
        gain = total_profit/total_cost

    return total_profit, total_cost, gain

def calc_always_high_odds_profit(X):
    total_bets = len(X)
    total_profit = 0

    for i, row in enumerate(X):
        date = row[0]
        time = row[1]
        team = row[2]
        opponent = row[3]
        team_odds = float(row[4])
        opp_odds = float(row[5])
        match_type = row[6]
        winner = row[7]

        if team_odds < opp_odds:
            pred = '1'
            if pred == winner:
                total_profit += team_odds - 1
            else:
                total_profit -= 1
        else:
            pred = '0'
            if pred == winner:
                total_profit += opp_odds - 1
            else:
                total_profit -= 1

    gain = total_profit/total_bets

    return total_profit, total_bets, gain

def get_recommendation(probs, team, opponent, market_team_odds, market_opp_odds, print_out=True):

    cut = (1 - abs((1/market_team_odds) + (1/market_opp_odds)))/2
    market_prob_team = (1/market_team_odds) + cut
    market_prob_opp = (1/market_opp_odds) + cut

    if probs[0] > market_prob_opp:
        bet_multiplier = (probs[0] - market_prob_opp)
        bet_team = opponent
    else:
        bet_multiplier = (probs[1] - market_prob_team)
        bet_team = team

    if print_out:
        print('Bet on : {0} with bet_muliplier = {1:0.4f}\n'.format(bet_team, bet_multiplier))


    return bet_multiplier, bet_team

if __name__ == '__main__':

    model = CatBoostClassifier()
    model.load_model('data/cat_model_v1')

    with open('data/df_v1.pkl', 'rb') as f:
        df = pickle.load(f)

    client = MongoClient()
    db = client['csgo']
    ranking_coll = db['team_stats_90d_ago']
    odds_coll = db['formated_odds_extra']


    X = np.array([
        ['2018-1-16', '9:00', 'fragsters', 'mainstreet', 1.285, 3.710, '', 1],
        ['2018-1-16', '9:00', 'demise', 'ambush', 1.757, 2.090, '', 0],
        ['2018-1-16', '9:00', 'extatus', 'aaa', 1.507, 2.6, 'bo3', 1],
        ['2018-1-16', '11:00', 'kinguin', 'windigo', 1.854, 1.970, 'bo3', 0],
        ['2018-1-15', '10:30', 'faze', 'natusvincere', 1.5, 2.620, '', 1],
        ['2018-1-15', '10:45', 'mousesports', 'spacesoldiers', 1.454, 2.78, '', 0],
        ['2018-1-15', '12:00', 'renegades', 'sprout', 1.526, 2.55, '', 1],
        ['2018-1-15', '13:15', 'envyus', 'quantumbellatorfire', 1.555, 2.469, '', 0],
        ['2018-1-16', '8:00', 'northacademy', 'redreserve', 1.909, 1.909, 'bo3', 1],
        ['2018-1-17', '9:00', 'lpsp', 'odense', 1.289, 3.68, 'bo3', 1],
        ['2018-1-17', '9:00', 'goldengitte', 'serenic', 1.222, 4.359, '', 1],
        ['2018-1-16', '21:00', 'candycamels', 'signature', 2.85, 1.4, 'bo3', 0],
        ['2018-1-17', '5:00', 'valiance', 'spirit', 1.649, 2.27, 'bo3', 0],
        ['2018-1-17', '8:00', 'enyoy', 'mansnothot', 2.160, 1.709, 'bo3', 0],
        ['2018-1-17', '11:00', 'ago', 'pride', 1.218, 4.400, 'bo3', 1],
        ['2018-1-17', '8:00', 'tricked', 'alternateattax', 1.625, 2.310, 'bo3', 0],
        ['2018-1-18', '5:00', 'pride', 'spirit', 1.588, 2.389, 'bo3', 0],
        ['2018-1-18', '8:00', 'kinguin', 'enyoy', 1.518, 2.570, 'bo3', 1],
        ['2018-1-18', '9:00', 'fragsters', 'lpsp', 1.645, 2.270, 'bo3', 0],
        ['2018-1-18', '11:00', 'extatus', 'kinguin', 1.662, 2.240, 'bo3', 0],
        ['2018-1-19', '7:00', 'liquid', 'big', 1.719, 2.150, '', 1],
        ['2018-1-19', '7:00', 'virtus.pro', 'quantumbellatorfire', 1.278, 3.770, '', 0],
        ['2018-1-19', '9:00', 'natusvincere', 'gambit', 1.934, 1.884, '', 0],
        ['2018-1-19', '10:00', 'astralis', 'mousesports', 1.584, 2.4, '', 0],
        ['2018-1-19', '11:00', 'north', 'vegasquadron', 1.555, 2.469, '', 0],
        ['2018-1-19', '12:00', 'skgaming', 'spacesoldiers', 1.318, 3.46, '', 1],
        ['2018-1-19', '13:00', 'faze', 'fnatic', 1.366, 3.170, '', 1],
        ['2018-1-19', '14:00', 'g2', 'cloud9', 1.699, 2.179, '', 1],
        ['2018-1-19', '17:00', 'order', 'athletico', 1.431, 2.87, 'bo3', 1],
        ['2018-1-19', '20:00', 'taintedminds', 'corvidae', 1.299, 3.6, 'bo3', 1],
        ['2018-1-20', '2:00', 'alternateattax', 'havu', 1.584, 2.4, 'bo3', 1],
        ['2018-1-20', '7:00', 'gambit', 'quantumbellatorfire', 1.160, 5.39, '', 0],
        ['2018-1-20', '8:15', 'cloud9', 'spacesoldiers', 1.531, 2.53, '', 0],
        ['2018-1-20', '9:30', 'faze', 'vegasquadron', 1.330, 3.38, '', 1],
        ['2018-1-20', '10:45', 'g2', 'liquid', 1.44, 2.83, '', 1],
        ['2018-1-20', '12:00', 'skgaming', 'mousesports', 1.588, 2.389, '', 1]
    ])

    total_profit, total_cost, gain = calc_profit_v2(model, df, X, ranking_coll)

    print('Using model\nTotal profit : ${0:0.2f}\nTotal cost : ${1:0.2f}\nGain : {2:0.4f}'.format(total_profit, total_cost, gain))

    # print('Using model and threshold strategy\nTotal profit : ${0:0.2f}\nTotal cost : ${1:0.2f}\nGain : {2:0.4f}'.format(total_profit_mp, total_cost_mp, gain_mp))
    #
    profit_ho, cost_ho, gain_ho = calc_always_high_odds_profit(X)

    print('Always betting on higher odds\nTotal profit : ${0:0.2f}\nTotal cost : ${1:0.2f}\nGain : {2:0.4f}'.format(profit_ho, cost_ho, gain_ho))


    ## Simulating profit with new data points:
    # X = [dates, times, teams, opponents, team_odds, opp_odds, match_types, winner]

    X = np.array([
        ['2018-1-16', '9:00', 'fragsters', 'mainstreet', 1.285, 3.710, '', 1],
        ['2018-1-16', '9:00', 'demise', 'ambush', 1.757, 2.090, '', 0],
        ['2018-1-16', '9:00', 'extatus', 'aaa', 1.507, 2.6, 'bo3', 1],
        ['2018-1-16', '11:00', 'kinguin', 'windigo', 1.854, 1.970, 'bo3', 0],
        ['2018-1-15', '10:30', 'faze', 'natusvincere', 1.5, 2.620, '', 1],
        ['2018-1-15', '10:45', 'mousesports', 'spacesoldiers', 1.454, 2.78, '', 0],
        ['2018-1-15', '12:00', 'renegades', 'sprout', 1.526, 2.55, '', 1],
        ['2018-1-15', '13:15', 'envyus', 'quantumbellatorfire', 1.555, 2.469, '', 0],
        ['2018-1-16', '8:00', 'northacademy', 'redreserve', 1.909, 1.909, 'bo3', 1],
        ['2018-1-17', '9:00', 'lpsp', 'odense', 1.289, 3.68, 'bo3', 1],
        ['2018-1-17', '9:00', 'goldengitte', 'serenic', 1.222, 4.359, '', 1],
        ['2018-1-16', '21:00', 'candycamels', 'signature', 2.85, 1.4, 'bo3', 0],
        ['2018-1-17', '5:00', 'valiance', 'spirit', 1.649, 2.27, 'bo3', 0],
        ['2018-1-17', '8:00', 'enyoy', 'mansnothot', 2.160, 1.709, 'bo3', 0],
        ['2018-1-17', '11:00', 'ago', 'pride', 1.218, 4.400, 'bo3', 1],
        ['2018-1-17', '8:00', 'tricked', 'alternateattax', 1.625, 2.310, 'bo3', 0],
        ['2018-1-18', '5:00', 'pride', 'spirit', 1.588, 2.389, 'bo3', 0],
        ['2018-1-18', '8:00', 'kinguin', 'enyoy', 1.518, 2.570, 'bo3', 1],
        ['2018-1-18', '9:00', 'fragsters', 'lpsp', 1.645, 2.270, 'bo3', 0],
        ['2018-1-18', '11:00', 'extatus', 'kinguin', 1.662, 2.240, 'bo3', 0],
        ['2018-1-19', '7:00', 'liquid', 'big', 1.719, 2.150, '', 1],
        ['2018-1-19', '7:00', 'virtus.pro', 'quantumbellatorfire', 1.278, 3.770, '', 0],
        ['2018-1-19', '9:00', 'natusvincere', 'gambit', 1.934, 1.884, '', 0],
        ['2018-1-19', '10:00', 'astralis', 'mousesports', 1.584, 2.4, '', 0],
        ['2018-1-19', '11:00', 'north', 'vegasquadron', 1.555, 2.469, '', 0],
        ['2018-1-19', '12:00', 'skgaming', 'spacesoldiers', 1.318, 3.46, '', 1],
        ['2018-1-19', '13:00', 'faze', 'fnatic', 1.366, 3.170, '', 1],
        ['2018-1-19', '14:00', 'g2', 'cloud9', 1.699, 2.179, '', 1],
        ['2018-1-19', '17:00', 'order', 'athletico', 1.431, 2.87, 'bo3', 1],
        ['2018-1-19', '20:00', 'taintedminds', 'corvidae', 1.299, 3.6, 'bo3', 1],
        ['2018-1-20', '2:00', 'alternateattax', 'havu', 1.584, 2.4, 'bo3', 1],
        ['2018-1-20', '7:00', 'gambit', 'quantumbellatorfire', 1.160, 5.39, '', 0],
        ['2018-1-20', '8:15', 'cloud9', 'spacesoldiers', 1.531, 2.53, '', 0],
        ['2018-1-20', '9:30', 'faze', 'vegasquadron', 1.330, 3.38, '', 1],
        ['2018-1-20', '10:45', 'g2', 'liquid', 1.44, 2.83, '', 1],
        ['2018-1-20', '12:00', 'skgaming', 'mousesports', 1.588, 2.389, '', 1]
    ])

    total_profit, total_cost, gain = calc_profit(model, df, X, ranking_coll)

    print('Using model\nTotal profit : ${0:0.2f}\nTotal cost : ${1:0.2f}\nGain : {2:0.4f}'.format(total_profit, total_cost, gain))

    total_profit_mp, total_cost_mp, gain_mp = calc_profit_model_probs(model, df, X, ranking_coll, threshold=.62)

    print('Using model and threshold strategy\nTotal profit : ${0:0.2f}\nTotal cost : ${1:0.2f}\nGain : {2:0.4f}'.format(total_profit_mp, total_cost_mp, gain_mp))

    profit_ho, cost_ho, gain_ho = calc_always_high_odds_profit(X)

    print('Always betting on higher odds\nTotal profit : {0:0.2f}\nTotal cost : {1:0.2f}\nGain : {2:0.4f}'.format(profit_ho, cost_ho, gain_ho))
