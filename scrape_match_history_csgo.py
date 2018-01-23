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
from pymongo import MongoClient
from fake_useragent import UserAgent

def wait():
    number = random.randint(2, 4) + random.random()
    time.sleep(number)
    print('Sleeping : {} secs'.format(number))

def show_elapsed_time(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    print("{}:{:>02}:{:>05.2f}".format(h, m, s))

def init_driver(useragent):

    #Using Firefox driver:
    profile = webdriver.FirefoxProfile()
    profile.set_preference("general.useragent.override", useragent.firefox)
    firefox_path = r'/Users/ryancoleman/Desktop/geckodriver'

    driver = webdriver.Firefox(firefox_profile=profile, executable_path=firefox_path)

    driver.implicitly_wait(120)


    return driver

def scrape_page(driver, mh_coll, first_page=False):
    # match_result_css_sel = '.results-all > .results-sublist'
    # match_result_eles = driver.find_elements_by_css_selector(match_result_css_sel)

    table_class_name = 'results-sublist'
    match_result_eles = driver.find_elements_by_class_name(table_class_name)

    wait()

    if first_page:
        match_result_eles = match_result_eles[1:]

    for ele in match_result_eles:
        day_match_result = ele.text.split('\n')
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

            mh_coll.update_one({'date' : date, 'team1' : team1, 'team2' : team2, 'team1_score' : team1_score, 'team2_score' : team2_score, 'event' : event, 'match_type' : match_type}, {'$set' : {'new_entry' : 'no'}}, upsert=True)

    # Check if we're at the last page:
    # pagination_css = '.pagination-data'
    # page_range, last_page = driver.find_element_by_css_selector(pagination_css).text.split('of')
    # wait()
    # start_range, end_range = page_range.split('-')
    # end_range = int(end_range)
    # last_page = int(last_page)

    last_page = 'last'
    end_range = 'end'

    return last_page, end_range

if __name__ == '__main__':

    client = MongoClient()
    db = client['csgo']
    mh_coll = db['match_history_format2_v1']

    start_time = time.time()
    ua = UserAgent()

    driver = init_driver(ua)

    url = 'https://www.hltv.org/results'
    driver.get(url)
    wait()

    not_last_page = True

    last_page, end_range = scrape_page(driver, mh_coll, first_page=True)
    page = 1
    print('Page : {}'.format(page))
    while not_last_page:
        url = 'https://www.hltv.org/results?offset={0}'.format(last_page)
        driver.get(url)
        wait()
        last_page, end_range = scrape_page(driver, mh_coll)
        page += 1
        print('Page : {}'.format(page))

        end_time = time.time()
        elapsed_time = end_time - start_time
        show_elapsed_time(elapsed_time)

        if end_range == last_page:
            not_last_page = False

    driver.quit()
