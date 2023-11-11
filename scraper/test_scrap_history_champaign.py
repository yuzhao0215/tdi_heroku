import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time

pd.options.display.max_columns = None
pd.options.display.max_rows = None
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

from ediblepickle import checkpoint


@checkpoint(
    key=lambda args, kwargs: args[-1],
    work_dir='./cache/every_day', refresh=False)
def get_df_day(url_, driver_, row_data_dic_, cache_name_):
    driver_.get(url_)

    table_element = WebDriverWait(driver_, 60). \
        until(
        EC.
            visibility_of_all_elements_located((By.XPATH, '//tr[@class="ng-star-inserted"]')))

    page_source = driver_.page_source

    soup = BeautifulSoup(page_source, 'lxml')

    tbody_elements = soup.findAll('tbody', class_='ng-star-inserted')
    print('lengh of tbodys: {}'.format(len(tbody_elements)))

    for i, el in enumerate(tbody_elements):
        row_elements = el.findAll('tr')

        for rw in row_elements:
            attri_name = rw.find('th').text

            # remove parenthesis
            attri_name = re.sub(r"\(.*\)", '', attri_name)

            # remove spaces at left and right
            attri_name = attri_name.strip()

            if attri_name in row_data_dic_:
                row_data_dic_[attri_name] = rw.find('td', class_='ng-star-inserted').text

    df = pd.DataFrame(row_data_dic_, index=[0])
    return df


@checkpoint(
    key=lambda args, kwargs: '_'.join([args[1], args[2], args[3].strftime("%m-%d-%Y"), args[4].strftime("%m-%d-%Y")]),
    work_dir='./cache', refresh=False)
def get_df_time_range(base_url, state_, city, start_day, end_day, driver_, sleeping_seconds=5):
    df_prep = pd.DataFrame()

    columns = ['date', 'High Temp', 'Low Temp', 'Day Average Temp',
               'Precipitation', 'Dew Point', 'High',
               'Low', 'Average', 'Max Wind Speed', 'Visibility', 'Sea Level Pressure',
               'Actual Time']

    while start_day != end_day:
        row_data = {c: np.nan for c in columns}
        row_data['date'] = start_day.strftime("%m/%d/%Y")
        # names = ['date']
        # values = [start_day.strftime("%m/%d/%Y")]

        print('gathering data from: ', start_day)
        formatted_lookup_URL = base_url.format(state_, city, start_day.year, start_day.month, start_day.day)

        day_cache_name = '_'.join([state_, city, start_day.strftime("%m-%d-%Y")])
        temp_df = get_df_day(formatted_lookup_URL, driver_, row_data, day_cache_name)
        df_prep = pd.concat([
            df_prep,
            temp_df
        ], ignore_index=True)

        start_day += timedelta(days=1)
        # print(formatted_lookup_URL)
        time.sleep(sleeping_seconds)
    return df_prep


# Use .format(YYYY, M, D)
lookup_URL = 'https://www.wunderground.com/history/daily/us/{}/{}/date/{}-{}-{}'
start_date = date(1980, 1, 1)
# end_date = start_date + pd.Timedelta(days=2)
end_date = date(1981, 1, 1)

state = 'il'
county = 'urbana'

# df_prep = pd.DataFrame()

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')

driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=options)

df = get_df_time_range(lookup_URL, state, county, start_date, end_date, driver)

print(df.columns)
print(df.values)
print(df.values.shape)