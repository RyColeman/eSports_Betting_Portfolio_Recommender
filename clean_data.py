import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

def get_stable_teams(df):
    unique_teams = df['team'].unique()
    team_wr_ind = list(df.columns).index('team_win_ratio')
    team_var_list = []
    for team in unique_teams:
        X = df[df['team'] == team].values
        total_games = len(X)
        win_ratios = X[:, team_wr_ind]
        team_wr_var = win_ratios.var()
        team_var_list.append((team, team_wr_var, total_games))

    team_var_list = [(team, var, tg) for team, var, tg in team_var_list if var !=0]
    team_var_list = sorted(team_var_list, key= lambda tup : tup[1])


    return team_var_list

def clean_match_type(mt):
    if mt == 'bo3' or mt == 'bo5' or mt == 'bo2':
        return mt
    else:
        return ''

def winner(team_score, opp_score):
    if team_score > opp_score:
        return 1
    elif team_score < opp_score:
        return 0
    else:
        return 'tie'

def make_dummies(df, col_name):
    dummies = pd.get_dummies(df[col_name])
    df = pd.concat([df, dummies], axis=1)
    df.drop(col_name, axis=1, inplace=True)
    return df

def get_doc_from_coll(date, coll):
    c = coll.find({'date' : date})
    for d in c:
        r = d
    return r['rankings']

def fix_team_names(team):
    if team == 'sk':
        return 'skgaming'
    if team == 'b.o.o.t-d[s' or team == 'b.o.o.t':
        return 'b.o.o.t-d[s]'
    if team == 'virtuspro':
        return 'virtus.pro'
    if team == 'teamliquid':
        return 'liquid'
    if team == 'agogaming':
        return 'ago'
    if team == 'bqfire':
        return 'quantumbellatorfire'
    return team

def add_schedule_strength(df, days1, days2):
    ''' days is an int'''
    # resetting df index:
    df.reset_index(drop=True, inplace=True)

    unique_teams = df['team'].unique()
    team_schedule_strength = np.zeros((len(df), 1))
    opp_schedule_strength = np.zeros((len(df), 1))
    team_opp_wins = np.zeros((len(df), 1))
    opp_opp_wins = np.zeros((len(df), 1))
    team_opp_losses = np.zeros((len(df), 1))
    opp_opp_losses = np.zeros((len(df), 1))
    team_wins_ratio = np.zeros((len(df), 1))
    opp_wins_ratio = np.zeros((len(df), 1))

    for team in unique_teams:
        df_team = df[df['team'] == team]
        team_ind = df_team.index
        X_team = df_team.values

        for i, row in enumerate(X_team):
            date = row[0]
            end_date1 = date - pd.Timedelta('{}d'.format(days1))
            X_range1_team = X_team[np.where((date > X_team[:,0]) & (X_team[:,0] > end_date1))]
            total_opp_games = X_range1_team[:, 11].sum()
            total_opp_wins = X_range1_team[:, 14].sum()
            total_opp_losses = total_opp_games - total_opp_wins
            team_opp_wins[team_ind[i]] = total_opp_wins
            team_opp_losses[team_ind[i]] = total_opp_losses

            end_date2 = date - pd.Timedelta('{}d'.format(days2))
            X_range2_team = X_team[np.where((date > X_team[:,0]) & (X_team[:,0] > end_date2))]

            if (i+1 == len(X_team)) or (len(X_range2_team) == 0):
                team_wins_ratio[team_ind[i]] = 0
            else:
                team_wins_past = X_range2_team[0, 13] - X_range2_team[-1, 13]
                team_games_past = X_range2_team[0, 10] - X_range2_team[-1, 10]
                if team_games_past == 0:
                    team_wins_ratio[team_ind[i]] = 0
                else:
                    team_wins_ratio[team_ind[i]] = team_wins_past/team_games_past

        df_opp = df[df['opponent'] == team]
        opp_ind = df_opp.index
        X_opp = df_opp.values

        for j, row in enumerate(X_opp):
            date = row[0]
            end_date1 = date - pd.Timedelta('{}d'.format(days1))
            X_range1_opp = X_opp[np.where((date > X_opp[:,0]) & (X_opp[:,0] > end_date1))]
            total_team_games = X_range1_opp[:, 10].sum()
            total_team_wins = X_range1_opp[:, 13].sum()
            total_team_losses = total_team_games - total_team_wins
            opp_opp_wins[opp_ind[j]] = total_team_wins
            opp_opp_losses[opp_ind[j]] = total_team_losses

            end_date2 = date - pd.Timedelta('{}d'.format(days2))
            X_range2_opp = X_opp[np.where((date > X_opp[:,0]) & (X_opp[:,0] > end_date2))]

            if (j+1 == len(X_opp)) or (len(X_range2_opp) == 0):
                opp_wins_ratio[opp_ind[j]] = 0
            else:
                opp_wins_past = X_range2_opp[0, 14] - X_range2_opp[-1, 14]
                opp_games_past = X_range2_opp[0, 11] - X_range2_opp[-1, 11]
                if opp_games_past == 0:
                    opp_wins_ratio[opp_ind[j]] = 0
                else:
                    opp_wins_ratio[opp_ind[j]] = opp_wins_past/opp_games_past

    team_schedule_strength = [w[0]/l[0] if l[0] != 0 else 0 for w, l in zip(team_opp_wins, team_opp_losses)]
    opp_schedule_strength = [w[0]/l[0] if l[0] != 0 else 0 for w, l in zip(opp_opp_wins, opp_opp_losses)]

    df.drop(['team_game_count', 'opp_game_count', 'team_wins', 'opp_wins'], axis=1, inplace=True)

    df['team_ss'] = team_schedule_strength
    df['opp_ss'] = opp_schedule_strength
    df['team_win_ratio'] = team_wins_ratio
    df['opp_win_ratio'] = opp_wins_ratio

    return df

def get_df():

    client = MongoClient()
    db = client['csgo']
    matches_coll = db['match_history_format2_v1']

    r = []
    c = matches_coll.find()
    for doc in c:
        r.append(doc)

    df = pd.DataFrame(r)
    df.drop(['_id', 'new_entry', 'event'], axis=1, inplace=True)

    # changing some team names to match up with all data:
    df.loc[:, 'team1'] = df['team1'].apply(lambda team : fix_team_names(team))
    df.loc[:, 'team2'] = df['team2'].apply(lambda team : fix_team_names(team))

    # Adding day of week column:
    # Mon = 0 -> Sun = 6
    df.loc[:, 'day_of_week'] = df['date'].apply(lambda date : date.dayofweek)

    # Adding month of year column:
    df.loc[:, 'month'] = df['date'].apply(lambda date : date.month)

    # Adding year column:
    df.loc[:, 'year'] = df['date'].apply(lambda date : date.year)

    # Adding day of month column:
    df.loc[:, 'day_of_month'] = df['date'].apply(lambda date : date.day)

    # Formatting dataframe such that team1 and team2 is team vs. Opponent.
    df1 = df.rename(columns={'team1': 'team', 'team2': 'opponent', 'team1_score': 'team_score', 'team2_score' : 'opp_score'})
    df2 = df.rename(columns={'team2': 'team', 'team1': 'opponent', 'team2_score': 'team_score', 'team1_score' : 'opp_score'})
    df = pd.concat([df1, df2])

    df.sort_index(inplace=True)

    # Adding total games played column:
    df['team_game_count'] = df.groupby('team').cumcount(ascending=False) + 1
    df['opp_game_count'] = df.groupby('opponent').cumcount(ascending=False) + 1

    # Adding Winner column:
    team_scores = df['team_score'].values
    opp_scores = df['opp_score'].values
    winners = [winner(ts, os) for ts, os in zip(team_scores, opp_scores)]
    winner_series = pd.Series(winners)
    df.loc[:, 'winner'] = winner_series.values

    #Filtering out tie games:
    df = df[df['winner'] != 'tie']

    df['winner'] = df['winner'].apply(lambda num : int(num))
    df_reverse = df.iloc[::-1]
    team_wins = df_reverse.groupby('team').cumsum()['winner']
    opp_wins = df_reverse.groupby('opponent').cumsum()['winner']
    # team_wins = team_wins.apply(lambda g : 0 if g == -1 else g)
    # opp_wins = opp_wins.apply(lambda g : 0 if g == -1 else g)
    df['team_wins'] = team_wins.iloc[::-1]
    df['opp_wins'] = opp_wins.iloc[::-1]

    # Making sure the date section is officially a pandas datetime:
    df['date'] = df['date'].apply(lambda d : pd.to_datetime(d))

    # Making sure match type is '', or bo2/3/5:
    df['match_type'] = df['match_type'].apply(lambda mt : clean_match_type(mt))

    df = add_schedule_strength(df, 90, 90)

    # getting rid of edge case team names that throw errors in this code pipeline:
    df = df[(df['team'] != 'g') & (df['opponent'] != 'g')]
    df = df[(df['team'] != 'dd') & (df['opponent'] != 'dd')]
    df = df[(df['team'] != '') & (df['opponent'] != '')]
    df = df[(df['team'] != 'team') & (df['opponent'] != 'team')]

    return df

def trim_df_to_max_teams(df, max_games):
    df_group = df.groupby('team').count()
    teams_max = df_group[df_group['date'] >= max_games].index

    df = df[df['team'].isin(teams_max)]

    return df

def add_ranking_features(df, coll):
    rank_version = []
    maps_played_team = []
    KD_diff_team = []
    KD_team = []
    ranking_team = []
    maps_played_opp = []
    KD_diff_opp = []
    KD_opp = []
    ranking_opp = []
    for i in range(len(df)):
        date = df['date'].iloc[i]
        team = df['team'].iloc[i]
        opp = df['opponent'].iloc[i]
        r = get_doc_from_coll(date, coll)
        rank_version.append(float(r[1]))
        team_rank = [rank.lower() for rank in r if team in rank.lower().replace(' ', '')]
        opp_rank = [rank.lower() for rank in r if opp in rank.lower().replace(' ', '')]
        if team_rank == []:
            maps_played_team.append(np.nan)
            KD_diff_team.append(np.nan)
            KD_team.append(np.nan)
            ranking_team.append(np.nan)
        else:
            team_rank = team_rank[0].split(' ')[-4:]
            maps_played_team.append(int(team_rank[0]))
            KD_diff_team.append(int(team_rank[1]))
            KD_team.append(float(team_rank[2]))
            ranking_team.append(float(team_rank[3]))
        if opp_rank == []:
            maps_played_opp.append(np.nan)
            KD_diff_opp.append(np.nan)
            KD_opp.append(np.nan)
            ranking_opp.append(np.nan)
        else:
            opp_rank = opp_rank[0].split(' ')[-4:]
            maps_played_opp.append(int(opp_rank[0]))
            KD_diff_opp.append(int(opp_rank[1]))
            KD_opp.append(float(opp_rank[2]))
            ranking_opp.append(float(opp_rank[3]))

    df.loc[:, 'rank_version'] = rank_version
    df.loc[:, 'maps_played_team'] = maps_played_team
    df.loc[:, 'maps_played_opp'] = maps_played_opp
    df.loc[:, 'KD_diff_team'] = KD_diff_team
    df.loc[:, 'KD_diff_opp'] = KD_diff_opp
    df.loc[:, 'KD_team'] = KD_team
    df.loc[:, 'KD_opp'] = KD_opp
    df.loc[:, 'ranking_team'] = ranking_team
    df.loc[:, 'ranking_opp'] = ranking_opp

    #filter to only include teams that have ratings:
    # df = df[(df['maps_played_team'] != '') & (df['maps_played_opp'] != '')]

    return df

def add_odds_data(df, odds_coll):
    cursor = odds_coll.find()
    raw_odds_list = []

    for doc in cursor:
        raw_odds_list.append(doc)

    df_odds = pd.DataFrame(raw_odds_list)
    if 'new_entry' in df_odds.columns:
        df_odds.drop(['new_entry'], axis=1, inplace=True)

    # Adding match number column:
    # df_odds = add_match_num_col(df_odds)

    # Creating a date without time column to match up with the main df:
    df_odds.loc[:, 'date_no_time'] = df_odds['date'].apply(lambda date : pd.to_datetime('{0}-{1}-{2}'.format(date.year, date.month, date.day)))

    # Renaming date columns to be consistant with main df:
    df_odds = df_odds.rename(columns={'date':'date_time', 'date_no_time':'date'})

    # Renaming specific team names to match other data:
    # changing some team names to match up with all data:
    df_odds.loc[:, 'team1'] = df_odds['team1'].apply(lambda team : fix_team_names(team))
    df_odds.loc[:, 'team2'] = df_odds['team2'].apply(lambda team : fix_team_names(team))

    # Creating an hour and minute column for when match took place:
    df_odds.loc[:, 'hour'] = df_odds['date_time'].apply(lambda date: date.hour)
    df_odds.loc[:, 'min'] = df_odds['date_time'].apply(lambda date: date.minute)

    # Making sure the date data here is a pandas datetime:
    df_odds['date'] = df_odds['date'].apply(lambda d : pd.to_datetime(d))

    # Reformating df_odds from team1 vs team2 to team vs. opponent:
    df_odds1 = df_odds.rename(columns={'team1':'team', 'team2': 'opponent', 'odds1':'team_odds', 'ah1':'team_ah', 'total1':'team_total', 'odds2':'opp_odds', 'ah2':'opp_ah', 'total2':'opp_total'})
    df_odds2 = df_odds.rename(columns={'team2':'team', 'team1': 'opponent', 'odds2':'team_odds', 'ah2':'team_ah', 'total2':'team_total', 'odds1':'opp_odds', 'ah1':'opp_ah', 'total1':'opp_total'})
    df_odds = pd.concat([df_odds1, df_odds2])

    # renaming map column to match up with main df for merging:
    # df_odds = df_odds.rename(columns={'map' : 'match_type'})

    # Filtering for bet types where match_type & details are both blank , '':
    df_odds = df_odds[(df_odds['map'] == '') & (df_odds['details'] == '')]

    df_odds.drop(['_id', 'date_time', 'details', 'draw_odds', 'event', 'map', 'opp_ah', 'opp_total', 'team_ah', 'team_total', 'page', 'time'], axis=1, inplace=True)

    df = pd.merge(df, df_odds, how='left', on=['team', 'opponent', 'date'])

    # Drop all rows that are NaN:
    df = df.dropna(axis=0, how='any')
    df['team_odds'] = df['team_odds'].apply(lambda to: to if to != '' else np.nan)
    df['opp_odds'] = df['opp_odds'].apply(lambda to: to if to != '' else np.nan)

    # Filtering for rows that have odds:
    # df = df[(df['team_odds'] != '') & (df['opp_odds'] != '')]

    return df

def grab_final_df(ranking_coll_name, max_games=100):
    df = get_df()

    # Trim df to include only teams that played a min amount of games throughout history:
    print('start trim_df_to_max_teams')
    df = trim_df_to_max_teams(df, max_games)
    print('end trim_df_to_max_teams')
    print('\n')

    # Adding odds data to matches
    client = MongoClient()
    db = client['csgo']
    odds_coll = db['formatted_odds_v6']
    df = add_odds_data(df, odds_coll)

    # Connecting to MongoDB for team ranking data:
    ranking_coll = db[ranking_coll_name]

    # Adding ranking features:
    print('start add_ranking_features')
    df = add_ranking_features(df, ranking_coll)
    print('end add_ranking_features')
    print('\n')

    # Experimenting with taking out teams with high winning rate volitility to see if this helps prediction strength:
    # team_var_list = get_stable_teams(df)
    # stable_teams = [team for team, var in team_var_list[:150]]
    # df = df[(df['team'].isin(stable_teams)) & (df['opponent'].isin(stable_teams))]

    df.drop(['opp_score', 'team_score'], axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    # Now drop all duplicate rows:
    df.drop_duplicates(inplace=True)

    # imputed_odds = df[['team_odds', 'opp_odds']].values
    # imputer = Imputer(strategy='most_frequent')
    # imputed_odds = imputer.fit_transform(imputed_odds)
    #
    # df['team_odds'] = imputed_odds[:, 0]
    # df['opp_odds'] = imputed_odds[:, 1]

    return df

if __name__ == '__main__':
    df = get_df()
    team = 'bravado'
    win_ratios = df[df['team'] == team]['team_win_ratio'].values[::-1]
    dates_x = df[df['team'] == team]['date'].values[::-1]
    dates_x = [pd.to_datetime(d).year for d in dates_x]
    xs = range(len(dates_x))

    def format_fn(tick_val, tick_pos):
        if int(tick_val) in xs:

            return dates_x[int(tick_val)]
        else:
            return ''

    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(xs[::-20])
    ax.plot(xs, win_ratios, linestyle='-', marker='D', markersize=6, markerfacecolor='g')
    ax.set_xlabel('Time')
    ax.set_ylabel('Win Rate (past 90day window)')
    ax.set_title("Team Bravado's Win Rate over time")
    plt.show()



    df = grab_final_df('team_stats_90d_ago', max_games=1)
    winner = df.pop('winner').values
    df['winner'] = winner

    sns.set(style="white")
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(10, 10))
    ax.figure.subplots_adjust(bottom = 0.2)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot=True, annot_kws={'size': 6}, cbar_kws={"shrink": .5})
    plt.show()
