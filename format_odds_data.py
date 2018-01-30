import numpy as np
import pandas as pd
import re
from pymongo import MongoClient
from collections import defaultdict
import pdb

def mtn_time_zone(date, time):
    hours, mins = time.split(':')
    delta = pd.Timedelta('{}h, {}m'.format(hours, mins))
    time_zone_change = pd.Timedelta('7h')
    datetime = date + delta - time_zone_change

    date = pd.to_datetime('{0}-{1}-{2}'.format(datetime.year, datetime.month, datetime.day))

    return datetime, date

def process_raw_odds(formatted_odds_coll, raw_odds, today_date, year=None, last_mon=None, page='new_data'):
    matches = []
    for i, line in enumerate(raw_odds):

        if 'E Sports' in line and ('CS:GO' in line or 'Counter-Strike: Global Offensive' in line):
            event = re.sub(r'E Sports >', '', line).strip().split(' - ')[1]
            still_in_csgo_segment = True
            j = i

            while still_in_csgo_segment:
                odds1 = ''
                odds2 = ''
                draw_odds = ''
                ah1 = ''
                ah2 = ''
                total1 = ''
                total2 = ''

                j += 1
                day, mon = raw_odds[j].split(' ')
                j += 1
                team1_line = raw_odds[j].lower()
                details = ''
                match_map = ''
                if re.search(r'map (\d)', team1_line) != None:
                    match_map = re.findall(r'map (\d)', team1_line)[0]
                    team1_line = re.sub('map '+match_map, '', team1_line).strip().strip(')').strip('(')
                elif re.search(r'map(\d)', team1_line) != None:
                    match_map = re.findall(r'map(\d)', team1_line)[0]
                    team1_line = re.sub('map'+match_map, '', team1_line).strip().strip(')').strip('(')
                if '1st' in team1_line and 'round' in team1_line:
                    details = re.findall('1st[\w ]+[Rounds|Round|round|rounds]', team1_line)[0].strip()
                    team1_line = re.sub(details, '', team1_line)
                time = re.findall(r'^[\d:]+', team1_line)[0]
                team1 = re.sub(time, '', team1_line).strip().lower()
                j += 1
                team2_line = raw_odds[j].lower().replace('()', '')

                if details != '':
                    team2_line = re.sub(details, '', team2_line)
                if match_map != '':
                    team2_line = re.sub('map '+match_map, '', team2_line)
                    if re.search(r'map\d', team2_line) != 1:
                        team2_line = re.sub(r'map\d', '', team2_line)

                if len(re.findall(r' [-\d.]+@[\d.]+$', team2_line)) == 1:
                    ah1 = re.findall(r' [-\d.]+@[\d.]+$', team2_line)[0].strip()
                    team2 = re.sub(ah1, '', team2_line).lower().strip()
                    j += 1
                    ah2 = raw_odds[j]

                elif len(re.findall(r'[o|u]\d.+@\d.+$', team2_line)) == 1:
                    total1 = re.findall(r'[o|u]\d.+@\d.+$', team2_line)[0].strip()
                    team2 = re.sub(total1, '', team2_line).lower().strip()
                    j += 1
                    total2 = raw_odds[j]

                elif len(re.findall(r' [\d.]+$', team2_line)) == 1:
                    odds1 = re.findall(r'[\d.]+$', team2_line)[0]
                    team2 = re.sub(odds1, '', team2_line).strip().lower()
                    j += 1
                    odds2_line = raw_odds[j].lower()
                    if len(odds2_line.strip().split(' ')) == 1:
                        odds2 = odds2_line.strip()
                    elif len(re.findall(r'[o|u]\d.+@\d.+$', odds2_line)) == 1:
                        total1 = re.findall(r'[o|u]\d.+@\d.+$', odds2_line)[0].strip()
                        odds2 = re.sub(total1, '', odds2_line).strip()
                        j += 1
                        total2 = raw_odds[j]
                    else:
                        odds2, ah1 = odds2_line.split(' ')
                        j += 1
                        ah2_line = raw_odds[j]
                        if len(ah2_line.split(' ')) > 1:
                            ah2, total1 = ah2_line.split(' ')
                            j += 1
                            total2 = raw_odds[j]
                        else:
                            ah2 = ah2_line
                elif 'offline' in team2_line:
                    team2 = re.sub('offline', '', team2_line).lower().strip()
                else:
                    team2 = team2_line.strip().lower()
                    if j+1 < len(raw_odds):
                        if 'Draw' in raw_odds[j+1]:
                            j += 1
                            odds1_line = raw_odds[j].lower()
                            if 'offline' not in odds1_line:
                                odds1 = re.sub('draw', '', odds1_line).strip()
                                j += 1
                                odds2 = raw_odds[j]
                                j += 1
                                draw_odds = raw_odds[j]
                        elif raw_odds[j+1].isdigit():
                            j += 1
                            odds2_line = raw_odds[j]
                            ah1 = odds2_line.strip()
                            j += 1
                            ah2 = raw_odds[j].strip()

                if last_mon == 'Dec' and mon == 'Jan':
                    year += 1
                if last_mon != None and year != None:
                    date = pd.to_datetime(str(year)+'-'+mon+'-'+day)
                    last_mon = mon

                    datetime, date = mtn_time_zone(date, time)

                # Getting rid of () and spaces to make sure all teams are the same format as other data sources
                team1 = team1.strip(')').strip(')').strip('(').strip(')').strip('(').strip(')').strip('(').strip(')').strip('(').strip(')').strip('(').replace(' ', '').replace('csgo', '').lower()
                team2 = team2.strip(')').strip(')').strip('(').strip(')').strip('(').strip(')').strip('(').strip(')').strip('(').strip(')').strip('(').replace(' ', '').replace('csgo', '').lower()

                if odds1 != '':
                    odds1 = float(odds1)
                if odds2 != '':
                    odds2 = float(odds2)

                if today_date != 'old_data':
                    # hours, mins = time.split(':')
                    # delta = pd.Timedelta('{}h,{}m'.format(hours, mins))
                    # datetime = today_date + delta - pd.Timedelta('8h')
                    datetime, date = mtn_time_zone(today_date, time)

                # formatted_odds_coll.update_one({'date' : date, 'team1' : team1, 'team2' : team2, 'time' : time, 'event': event, 'details' : details, 'map': match_map, 'page' : page}, {'$set' : {'odds1' : odds1, 'odds2' : odds2, 'draw_odds' : draw_odds, 'ah1' : ah1, 'ah2' : ah2, 'total1' : total1, 'total2' : total2}}, upsert=True)

                formatted_odds_coll.update_one({'date' : date, 'team1' : team1, 'team2' : team2, 'time' : time, 'event': event, 'details' : details, 'map': match_map, 'page' : page, 'odds1' : odds1, 'odds2' : odds2, 'draw_odds' : draw_odds, 'ah1' : ah1, 'ah2' : ah2, 'total1' : total1, 'total2' : total2}, {'$set' : {'new_entry' : 'no'}}, upsert=True)

                if odds1 != '' and odds2 != '':
                    # adjusting time/date for correct timezone (P) that predict script uses:
                    # hours, mins = time.split(':')
                    # delta = pd.Timedelta('{}h,{}m'.format(hours, mins))
                    # adj_datetime = date + delta - pd.Timedelta('8h')
                    # adj_datetime = date - pd.Timedelta('1h')


                    match = {'team' : team1, 'opponent' : team2, 'team_odds' : odds1, 'opp_odds' : odds2, 'datetime' : datetime, 'bet_type' : match_map, 'details' : details}

                    matches.append(match)

                if raw_odds[j] == raw_odds[-1]:
                    still_in_csgo_segment = False
                elif 'TIME' in raw_odds[j+1]:
                    still_in_csgo_segment = False
                elif 'E Sports' in raw_odds[j+1]:
                    still_in_csgo_segment = False

    return matches, year, last_mon

if __name__ == '__main__':
    client = MongoClient()
    db = client['csgo']
    coll = db['odds_all_esports']
    formatted_odds_coll = db['formatted_odds_v6']

    # raw_odds_list = []
    c = coll.find()
    year = 2015
    last_mon = ''
    for doc in c:
        csgo_segment = False
        raw_odds = doc['odds_table']
        page = doc.get('page', 'new_data')
        today_date = doc.get('today_date', 'old_data')

        matches, year, last_mon = process_raw_odds(formatted_odds_coll, raw_odds, today_date, year=year, last_mon=last_mon, page=page)
