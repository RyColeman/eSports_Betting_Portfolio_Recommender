import pandas as pd
import numpy as np
import re
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
from catboost import CatBoostClassifier
import scrape_odds_csgo as scrape
import format_odds_data as fod
import predict as pt

def get_date_name(date):
    day = date.day
    month_name_dict = {1:'January', 2: 'February', 3: 'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
    month_name = month_name_dict[date.month]
    if day == 1:
        day_name = '1st'
    elif day == 2:
        day_name = '2nd'
    elif day == 3:
        day_name = '3rd'
    else:
        day_name = '{}th'.format(day)

    date_name = 'Results for {0} {1} {2}'.format(month_name, day_name, date.year)
    return date_name


def scrape_todays_odds(today_date, odds_coll):
    ua = UserAgent()

    driver = scrape.init_driver(ua, implicit_wait=20)
    url = 'https://www.oddsarchive.com/#stime=today&sport=E Sports'
    driver.get(url)
    scrape.wait()

    # esports_button_xpath ='''/html/body/div[1]/div[1]/section[2]/div[2]/ul/li[4]/div/span[1]'''
    esports_button_xpath = '''//span[text()="E Sports"]'''
    driver.find_element_by_xpath(esports_button_xpath).click()
    scrape.wait()

    next_page_link_text = '»'
    table_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div[2]/table/tbody'''

    odds_table = []

    full_table = driver.find_element_by_xpath(table_xpath).text.split('\n')
    scrape.wait()
    odds_table.extend(full_table)

    while scrape.link_text_element_exists(driver, next_page_link_text):

        driver.find_element_by_link_text(next_page_link_text).click()
        scrape.wait()

        full_table = driver.find_element_by_xpath(table_xpath).text.split('\n')
        odds_table.extend(full_table)
        scrape.wait()

    # Since data was originally scraped with page as a unique id, we'll add another column, date, to uniquely identify these new raw_odds entries in the same MongoDB collection:
    odds_coll.update_one({'today_date' : today_date}, {'$set' : {'odds_table' : odds_table}}, upsert=True)

    driver.quit()

    return odds_table

def get_recommendations(team, opponent, probs, market_probs, threshold):
    # For threshold strategy:
    thresh_strat = 'Do Not Bet'
    if probs[0] > threshold:
        thresh_strat = opponent
    elif probs[1] > threshold:
        thresh_strat = team

    # For Beating Odds Strategy #1:
    bo1_strat = 'Do Not Bet'
    if probs[0] > probs[1]:
        if probs[0] > market_probs[0]:
            bo1_strat = opponent
    else:
        if probs[1] > market_probs[1]:
            bo1_strat = team

    # For Beating Odds Strategy #2:
    if probs[0] > market_probs[0]:
        bo2_strat = opponent
    elif probs[1] > market_probs[1]:
        bo2_strat = team

    recs_dict = {'thresh_strat' : thresh_strat, 'bo1_strat' : bo1_strat, 'bo2_strat' : bo2_strat}

    return recs_dict

def scrape_yesterday_match_history(match_history_coll, yesterday_date):
    ua = UserAgent()
    driver = scrape.init_driver(ua, implicit_wait=20)
    ## Accounting for timezone difference in scraping source
    # timezone_change = pd.Timedelta('8h')
    # yesterday_datetime = yesterday_datetime + timezone_change
    year = yesterday_date.year
    month = yesterday_date.month
    day = yesterday_date.day

    if len(str(month)) == 1:
        month = '0'+str(month)
    if len(str(day)) == 1:
        day = '0'+str(day)

    url = 'https://www.hltv.org/results?startDate={0}-{1}-{2}&endDate={0}-{1}-{2}'.format(year, month, day)
    driver.get(url)
    scrape.wait()

    match_table_css = '.results-holder'

    match_result = driver.find_element_by_css_selector(match_table_css).text.split('\n')

    date_name = get_date_name(yesterday_date)
    prev_date_name = get_date_name(yesterday_date - pd.Timedelta('1d'))

    match_result = driver.find_element_by_css_selector(match_table_css).text.split('\n')
    scrape.wait()
    ind_start = match_result.index(date_name)
    ind_end = match_result.index(prev_date_name)

    day_match_result = match_result[ind_start:ind_end]

    date_line = day_match_result[0].split()
    month = date_line[2]
    day = re.findall(r'\d+', date_line[3])[0]
    year = date_line[-1]

    date = pd.to_datetime('{0}-{1}-{2}'.format(year, month, day))

    team1_list = day_match_result[1::5]
    score_list = day_match_result[2::5]
    team2_list = day_match_result[3::5]
    event_list = day_match_result[4::5]
    match_type_list = day_match_result[5::5]

    for i in range(len(team1_list)):
        team1 = team1_list[i].lower().replace(' ', '')
        team2 = team2_list[i].lower().replace(' ', '')
        team1_score, team2_score = score_list[i].split('-')
        team1_score = int(team1_score.strip())
        team2_score = int(team2_score.strip())
        event = event_list[i]
        match_type = match_type_list[i]

        match_history_coll.update_one({'date' : date, 'team1' : team1, 'team2' : team2, 'team1_score' : team1_score, 'team2_score' : team2_score, 'event' : event, 'match_type' : match_type}, {'$set' : {'new_entry' : 'no'}}, upsert=True)

    driver.quit()

class BetAssistant():
    def __init__(self):
        self.portfolio = []

        ## Using model that dropped all time/date information and dropped team/opponent schedule strength and win_ratios. (Unfortunately these features were not predictive).
        clf = CatBoostClassifier()
        clf.load_model('data/cat_model_v2')
        self._model = clf

        # unpicking the first df used to train/test the beta model:
        with open('data/df_v2.pkl', 'rb') as f:
            self._df = pickle.load(f)

        client = MongoClient()
        db = client['csgo']
        self._unformatted_odds_coll = db['odds_all_esports_test']
        self._formatted_odds_coll = db['formatted_odds_test']
        self._matches_coll = db['match_history_format2_v1']
        self._ranking_coll = db['team_stats_90d_ago']
        self._market_odds_coll = db['market_odds_data_v1']

        today_datetime = pd.Timestamp.today()
        self.today_date = pd.to_datetime('{0}-{1}-{2}'.format(today_datetime.year, today_datetime.month, today_datetime.day))
        self.eligible_matches = []

    def update_data(self):
        ''' scrape OA for new today's matches. (after 5pm MTN, gets tomorrow's matches). Inserts new odds data into local database. Generates probabilities for all these matches.'''

        yesterday_date = self.today_date - pd.Timedelta('1d')
        yesterday_datetime = pd.Timestamp.today() - pd.Timedelta('1d')

        # Checking if matches_coll already contains match history data from yesterday_date:
        r = []
        cursor = self._matches_coll.find({'date' : yesterday_date})
        for doc in cursor:
            r.append(doc)
        if r == []:
            # Adding yesterday's match results to our match history MongoDB collection:
            scrape_yesterday_match_history(self._matches_coll, yesterday_date)

        # See if today's odds are already in the database. If they aren't, then scrape:
        # adjusting today's date (MTN) to the odd source's timezone:
        today_datetime_adj = pd.Timestamp.today() + pd.Timedelta('7h')
        today_date_adj = pd.to_datetime('{0}-{1}-{2}'.format(today_datetime_adj.year, today_datetime_adj.month, today_datetime_adj.day))

        raw_odds = None
        cursor = self._unformatted_odds_coll.find({'today_date' : today_date_adj})
        for doc in cursor:
            raw_odds = doc['odds_table']

        if raw_odds is None:
            raw_odds = scrape_todays_odds(today_date_adj, self._unformatted_odds_coll)

        # Time to format the raw_odds into CSGO matches with odds (the year and last_mon aren't neccessary for this script, so we won't be using these variables):
        matches, year, last_mon = fod.process_raw_odds(self._formatted_odds_coll, raw_odds, today_date_adj)

        add_all = False
        if self.eligible_matches == []:
            add_all = True
        else:
            previously_added_matches = self.eligible_matches

        for match in matches:
            team = match['team']
            opponent = match['opponent']
            team_odds = match['team_odds']
            opp_odds = match['opp_odds']
            datetime = match['datetime']
            h = datetime.hour
            m = datetime.minute
            if len(str(m)) == 1:
                m = '0{}'.format(m)
            date = '{0}-{1}-{2}'.format(datetime.year, datetime.month, datetime.day)
            time = '{0}:{1}'.format(h, m)
            match_type = match['bet_type']
            details = match['details']

            if details == '' and match_type == '':

                probs, advice = pt.get_predictions_v2(self._model, self._df, date, time, team, opponent, team_odds, opp_odds, self._ranking_coll, match_type, print_out=False)

                should_I_bet = advice[0]
                if should_I_bet and pd.Timestamp.today() < datetime:
                    bet_match = {'team' : team, 'opponent' : opponent, 'team_prob' : probs[1], 'opp_prob' : probs[0], 'time' : time, 'date': date, 'team_odds' : team_odds, 'opp_odds' : opp_odds}

                    # first check if match is already in this list of eligible matches if not, add the match:
                    add_match = True
                    if add_all:
                        self.eligible_matches.append(bet_match)
                    else:
                        for prev_match in previously_added_matches:
                            h, m = time.split(':')
                            ph, pm = prev_match['time'].split(':')
                            if ((int(ph) == int(h) and int(pm) == int(m)) and ((prev_match['team'] == team and prev_match['opponent'] == opponent) or (prev_match['opponent'] == team and prev_match['team'] == opponent))):
                                add_match = False
                                break
                        if add_match:
                            self.eligible_matches.append(bet_match)

    def insert_games_data(self, games):
        ''' User manually enters in games that are taking place today on P. Function then generates probabilities for all these matches.
        Note: This function will only add matches where both teams were present in the model's training dataset and where both teams have stats data available for this given date.
        games --> [[date='YYYY-MM-DD', time='HH:MM', team='string', opponent='string', team_odds, opp_odds, match_type]]'''

        add_all = False
        if self.eligible_matches == []:
            add_all = True
        else:
            previously_added_matches = self.eligible_matches

        for date, time, team, opponent, team_odds, opp_odds, match_type in games:

            team = team.lower().replace(' ', '')
            opponent = opponent.lower().replace(' ', '')

            probs, advice = pt.get_predictions_v2(self._model, self._df, date, time, team, opponent, team_odds, opp_odds, self._ranking_coll, match_type, print_out=False)

            should_I_bet = advice[0]
            if should_I_bet:
                bet_match = {'team' : team, 'opponent' : opponent, 'team_prob' : probs[1], 'opp_prob' : probs[0], 'time' : time, 'date' : date, 'team_odds' : team_odds, 'opp_odds' : opp_odds}
                # first check if match is already in this list of eligible matches if not, add the match:
                add_match = True
                if add_all:
                    self.eligible_matches.append(bet_match)
                else:
                    for prev_match in previously_added_matches:
                        h, m = time.split(':')
                        ph, pm = prev_match['time'].split(':')
                        if (int(ph) == int(h) and int(pm) == int(m)) and ((prev_match['team'] == team and prev_match['opponent'] == opponent) or (prev_match['opponent'] == team and prev_match['team'] == opponent)):
                            add_match = False
                            break
                    if add_match:
                        self.eligible_matches.append(bet_match)

    def get_today_portfolio(self, odds_list, today_budget, market_name, print_out=True):
        ''' Function takes the odds of the given betting market in odds_list and today_budget to generate a betting portfolio that tells user which teams to put money on in each match and how much money to bet on each team.
        odds_list --> [[date='YYYY-MM-DD', 'team', 'opponent', team_odds, opp_odds]]
        today_budget --> float
        market_name --> string

        Note: Only games that were inserted manually and scrapped will be included. Furthermore, if any games include teams that were not included in model's training data or include teams were there were no stats data found, then these matches will not be included in today's portfolio.
        '''
        self.portfolio = []

        if self.eligible_matches == []:
            print("There's no matches to choose from. Run the 'update_data' method and/or insert some matches with the 'insert_games_data' method.")
        else:
            bm_sum = 0
            for date, team, opponent, market_team_odds, market_opp_odds in odds_list:

                team = team.lower().replace(' ', '')
                opponent = opponent.lower().replace(' ', '')

                for match in self.eligible_matches:
                    add_match = False
                    if team == match['team'] and opponent == match['opponent']:
                        team_prob = match['team_prob']
                        opp_prob = match['opp_prob']
                        add_match = True
                    elif team == match['opponent'] and opponent == match['team']:
                        team_prob = match['opp_prob']
                        opp_prob = match['team_prob']
                        add_match = True

                    if add_match:
                        probs = [opp_prob, team_prob]
                        bet_multiplier, bet_team = pt.get_recommendation(probs, team, opponent, market_team_odds, market_opp_odds, print_out=False)

                        bm_sum += bet_multiplier

                        self.portfolio.append({'bet_multiplier': bet_multiplier, 'bet_team' : bet_team, 'match' : match})

                        # Add new market_odds datapoints into local MongoDB:
                        h, m = match['time'].split(':')
                        delta = pd.Timedelta('{0}h,{1}m'.format(h, m))
                        datetime = pd.to_datetime(date) + delta

                        self._market_odds_coll.update_one({'datetime': datetime, 'team' : team, 'opponent' : opponent, 'team_odds' : market_team_odds, 'opp_odds' : market_opp_odds, 'market_name' : market_name}, {'$set' : {'new_entry' : 'no'}}, upsert=True)

            # Now we opptimize to find the right amount of money to put on each match:

            bet_const = 0.001
            payment = 0.001
            while (today_budget - payment) > 0.001:
                payment = bet_const*bm_sum
                bet_const += 0.01

            for game in self.portfolio:
                game['bet_amount'] = game['bet_multiplier']*bet_const

                if print_out:
                    print('Put ${0:0.2f} on {1} for this match :\n{2}\n'.format(game['bet_amount'], game['bet_team'], game['match']))

if __name__ == '__main__':

    ba = BetAssistant()
    # scraping for today's matches.
    ba.update_data()

    # Inserting additional matches that may not have been availble on scraping website source.
    # new_matches = [[date='YYYY-MM-DD', time='HH:MM', team='string', opponent='string', team_odds, opp_odds, match_type]]
    new_matches = [
    ['2018-1-22', '2:00', 'grayhound', 'chiefs', 1.414, 2.929, 'bo3'],
    ['2018-1-23', '17:30', 'mythic', 'maskoff', 1.395, 3.02, ''],
    ['2018-1-23', '18:30', 'ronin', 'orion', 1.158, 5.430, ''],
    ['2018-1-23', '19:30', 'swolepatrol', 'iceberg', 1.297, 3.61, '']
    ]

    ba.insert_games_data(new_matches)

    # Now enter the odds of the market you want to bet on and your day's budget:
    # [[date='YYYY-MM-DD', 'team', 'opponent', market_team_odds, market_opp_odds, market_name]]

    new_odds_list = [
    ['2018-1-22', 'chiefs', 'grayhound', 3.21, 1.29],
    ['2018-1-22', 'ehome', 'redwolf', 1.74, 1.86]
    ]
    today_budget = 100
    ba.get_today_portfolio(new_odds_list, today_budget, 'thunderpick')
