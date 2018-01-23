import time
import re
import random
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.select import Select
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

def init_driver(ua, implicit_wait=90):

    #Using Firefox driver:
    profile = webdriver.FirefoxProfile()
    profile.set_preference("general.useragent.override", ua.firefox)
    firefox_path = r'/Users/ryancoleman/Desktop/geckodriver'

    driver = webdriver.Firefox(firefox_profile=profile, executable_path=firefox_path)

    driver.implicitly_wait(implicit_wait)

    return driver

def link_text_element_exists(driver, link_text):
    if driver.find_elements_by_link_text(link_text) == []:
        return False
    else:
        return True

if __name__ == '__main__':

    client = MongoClient()
    db = client['csgo']
    coll = db['odds_all_esports']

    start_time = time.time()
    # example option: add 'incognito' command line arg to options
    ua = UserAgent()

    driver = init_driver(ua)
    search_url = 'https://www.oddsarchive.com/search'
    wait()
    driver.get(search_url)
    wait()

    time_bubble_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[2]/div/label[3]'''
    driver.find_element_by_xpath(time_bubble_xpath).click()
    wait()

    esports_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[1]/select[1]/option[14]'''
    driver.find_element_by_xpath(esports_xpath).click()
    wait()

    search_button_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[1]/input[2]'''
    driver.find_element_by_xpath(search_button_xpath).click()
    wait()

    ############# Version 1: loop through all eSports ########################
    page = 1
    odds_table_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/div[2]/div/table/tbody'''
    odds_table = driver.find_element_by_xpath(odds_table_xpath).text.split('\n')
    wait()
    coll.update_one({'page' : page}, {'$set' : {'odds_table' : odds_table}}, upsert=True)
    print('......................')
    print('Page : {}'.format(page))
    elapsed_time = time.time() - start_time
    show_elapsed_time(elapsed_time)
    # page += 1

    # Go to next page total 16922:

    next_page_link_text = '»'
    # While the next page button exists, then scrape.
    # while link_text_element_exists(driver, next_page_link_text):
    # 8946
    # 11204
    for page in range(12775, 16923):

        next_page_url = 'https://www.oddsarchive.com/search#s=1&stime=past&ssport=12&page={0}&page={0}'.format(page)
        driver.get(next_page_url)
        wait()
        # driver.find_element_by_link_text(next_page_link_text).click()
        # wait()
        odds_table = driver.find_element_by_xpath(odds_table_xpath).text.split('\n')
        wait()
        coll.update_one({'page' : page}, {'$set' : {'odds_table' : odds_table}}, upsert=True)
        print('......................')
        print('Page : {}'.format(page))
        elapsed_time = time.time() - start_time
        show_elapsed_time(elapsed_time)
        # page += 1

    driver.quit()

    ###################### Version 2: Specifically loop through CS:GO #########

    # page = 1
    # ######### Preparing to loop through all the different CS:GO leagues:
    # for option in range(3, 73):
    #
    #     # time_bubble_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[2]/div/label[3]'''
    #     # driver.find_element_by_xpath(time_bubble_xpath).click()
    #     # wait()
    #     #
    #     # esports_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[1]/select[1]/option[14]'''
    #     # driver.find_element_by_xpath(esports_xpath).click()
    #     # wait()
    #
    #     csgo_league_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[1]/select[2]/option[{}]'''.format(option)
    #     driver.find_element_by_xpath(csgo_league_xpath).click()
    #     wait()
    #
    #     search_button_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/form/div[1]/div[1]/input[2]'''
    #     driver.find_element_by_xpath(search_button_xpath).click()
    #     wait()
    #
    #     odds_table_xpath = '''/html/body/div[1]/div[1]/section[1]/div/div/div[2]/div/table/tbody'''
    #     odds_table = driver.find_element_by_xpath(odds_table_xpath).text.split('\n')
    #     wait()
    #     coll.update_one({'page' : page}, {'$set' : {'odds_table' : odds_table}}, upsert=True)
    #     print('......................')
    #     print('Leagues looped : {}'.format(option))
    #     print('Page : {}'.format(page))
    #     elapsed_time = time.time() - start_time
    #     show_elapsed_time(elapsed_time)
    #     page += 1
    #
    #     ## Find out how many pages are in this given CS:GO league category:
    #     # last_page_link_text = '»»'
    #     next_page_link_text = '»'
    #     # While the next page button exists, then scrape.
    #     # i = 2
    #     while link_text_element_exists(driver, next_page_link_text):
    #         # next_page_url = 'https://www.oddsarchive.com/search#s=1&stime=past&ssport=12&page={0}&page={0}'.format(i)
    #         # driver.get(next_page_url)
    #         driver.find_element_by_link_text(next_page_link_text).click()
    #         wait()
    #         odds_table = driver.find_element_by_xpath(odds_table_xpath).text.split('\n')
    #         wait()
    #         coll.update_one({'page' : page}, {'$set' : {'odds_table' : odds_table}}, upsert=True)
    #         print('......................')
    #         print('Leagues looped : {}'.format(option))
    #         print('Page : {}'.format(page))
    #         elapsed_time = time.time() - start_time
    #         show_elapsed_time(elapsed_time)
    #         page += 1
    #         # i += 1
    #
    #     # if link_text_element_exists(driver, last_page_link_text):
    #     #     driver.find_element_by_link_text(last_page_link_text).click()
    #     #     last_page_url = driver.current_url
    #     #     last_page = int(re.findall(r'[0-9]+$', last_page_url)[0])
    #     #
    #     #     for i in range(2, last_page+1):
    #     #         next_page_url = 'https://www.oddsarchive.com/search#s=1&stime=past&ssport=12&page={0}&page={0}'.format(i)
    #     #         driver.get(next_page_url)
    #     #         wait()
    #     #         odds_table = driver.find_element_by_xpath(odds_table_xpath).text.split('\n')
    #     #         wait()
    #     #         coll.update_one({'page' : page}, {'$set' : {'odds_table' : odds_table}}, upsert=True)
    #     #         print('......................')
    #     #         print('Page : {}'.format(page))
    #     #         elapsed_time = time.time() - start_time
    #     #         show_elapsed_time(elapsed_time)
    #     #         page += 1
    #     #
    #     # elif link_text_element_exists(driver, next_page_link_text):
    #     #     driver.find_element_by_link_text(next_page_link_text).click()
    #     #     last_page_url = driver.current_url
    #     #     last_page = int(re.findall(r'[0-9]+$', last_page_url)[0])
    #     #
    #     #     for i in range(2, last_page+1):
    #     #         next_page_url = 'https://www.oddsarchive.com/search#s=1&stime=past&ssport=12&page={0}&page={0}'.format(i)
    #     #         driver.get(next_page_url)
    #     #         wait()
    #     #         odds_table = driver.find_element_by_xpath(odds_table_xpath).text.split('\n')
    #     #         wait()
    #     #         coll.update_one({'page' : page}, {'$set' : {'odds_table' : odds_table}}, upsert=True)
    #     #         print('......................')
    #     #         print('Page : {}'.format(page))
    #     #         elapsed_time = time.time() - start_time
    #     #         show_elapsed_time(elapsed_time)
    #     #         page += 1
    #
    #
    #     # driver.quit()
    #     # driver = init_driver(ua)
    #     # wait()
    #     # driver.get(search_url)
    #     # wait()
    #
    # # driver.quit()
