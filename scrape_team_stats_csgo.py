import time
import re
import random
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from pymongo import MongoClient
from clean_data import get_df

def wait():
    number = random.randint(2, 4) + random.random()
    time.sleep(number)
    print('Sleeping : {} secs'.format(number))

def show_elapsed_time(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    print("{}:{:>02}:{:>05.2f}".format(h, m, s))

# def click_date(date):
#     year, month, day = date.year, date.month, date.day
#
#     delta = pd.Timedelta('30 days')
#     start_date = date - delta
#     start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
#
#     start_box_xpath = '/html/body/div[2]/div/div[2]/div[2]/div/div[4]/div[2]/div/div/div[4]/div/div/input[1]'
#     end_box_xpath = '/html/body/div[2]/div/div[2]/div[2]/div/div[4]/div[2]/div/div/div[4]/div/div/input[2]'
#
#     start_box = driver.find_element_by_xpath(start_box_xpath)
#     end_box = driver.find_element_by_xpath(end_box_xpath)
#
#     start_box.clear()
#     end_box.clear()
#
#     start_date_string = '{0}-{1}-{2}'.format(start_year, start_month, start_day)
#     end_date_string = '{0}-{1}-{2}'.format(year, month, day)
#
#     start_box.send_keys(start_date_string)
#     end_box.send_keys(end_date_string)
#     start_box.send_keys(start_date_string)

def init_driver(useragent):

    #Using Firefox driver:
    profile = webdriver.FirefoxProfile()
    profile.set_preference("general.useragent.override", useragent.firefox)
    firefox_path = r'/Users/ryancoleman/Desktop/geckodriver'

    driver = webdriver.Firefox(firefox_profile=profile, executable_path=firefox_path)

    driver.implicitly_wait(90)


    return driver

def change_url_date(driver, date, n):
    year, month, day = date.year, date.month, date.day

    delta = pd.Timedelta('{} days'.format(n))
    start_date = date - delta
    start_year, start_month, start_day = start_date.year, start_date.month, start_date.day

    if month < 10:
        month = '0'+str(month)
    if day < 10:
        day = '0'+str(day)
    if start_month < 10:
        start_month = '0'+str(start_month)
    if start_day < 10:
        start_day = '0'+str(start_day)

    start_date_string = '{0}-{1}-{2}'.format(start_year, start_month, start_day)
    end_date_string = '{0}-{1}-{2}'.format(year, month, day)

    url = 'https://www.hltv.org/stats/teams?minMapCount=0&startDate={0}&endDate={1}'.format(start_date_string, end_date_string)

    driver.get(url)

if __name__ == '__main__':
    start_time = time.time()
    # example option: add 'incognito' command line arg to options
    ua = UserAgent()
    n = 90

    df = get_df()
    dates_df = df['date'].unique()
    # dates = set(df['date'].unique())
    # dates = list(dates)
    dates_df = set([pd.to_datetime(date) for date in dates_df])

    # dates = [pd.to_datetime('2016-1-27'), pd.to_datetime('2016-12-24'), pd.to_datetime('2016-2-29'), pd.to_datetime('2016-3-6')]

    driver = init_driver(ua)
    url = 'https://www.hltv.org/stats/teams?minMapCount=0'
    driver.get(url)
    wait()

    # rankings_dict = {}
    rankings_xpath = '/html/body/div[2]/div/div[2]/div[1]/div/table'
    # team_start_dates = get_team_start_dates(df)
    # initiating a MongoDB client to store scraped data in a MongoDB database
    client = MongoClient()
    db = client['csgo']
    coll = db['team_stats_{}d_ago'.format(n)]

    # Grabbing a list of dates that haven't been updated yet:
    dates_coll = []
    c = coll.find()
    for doc in c:
        dates_coll.append(doc['date'])

    dates_coll = set([pd.to_datetime(d) for d in dates_coll])

    dates_to_update = dates_df.difference(dates_coll)

    for page, date in enumerate(dates_to_update):
        print('page : {}'.format(page))
        if page+1 % 100 == 0:
            url = driver.current_url
            driver.quit()
            driver = init_driver()
            driver.get(url)
            wait()
        # click_date(date)
        yesterday = date - pd.Timedelta('1 day')
        change_url_date(driver, yesterday, n)
        wait()
        try:
            rankings = driver.find_element_by_xpath(rankings_xpath).text.split('\n')
        except:
            wait()
            rankings = driver.find_element_by_xpath(rankings_xpath).text.split('\n')
        ## storing scraped info in a MongoDB database
        coll.update_one({'date' : date}, {'$set' : {'rankings' : rankings}}, upsert=True)
        wait()
        print('......................')
        elapsed_time = time.time() - start_time
        show_elapsed_time(elapsed_time)
    driver.quit()
