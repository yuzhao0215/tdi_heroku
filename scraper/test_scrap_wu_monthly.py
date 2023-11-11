import datetime

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
from dateutil.relativedelta import relativedelta


@checkpoint(
    key=lambda args, kwargs: '_'.join(map(str, [args[0], args[1], args[2], args[3], "wu"])),
    work_dir='./cache/county_weather_wu', refresh=True)
def get_dayily_weather_from_month_table(state_, county_, year_, month_, driver_):
    base_url = "https://www.wunderground.com/history/monthly/us/{}/{}/date/{}-{}"

    lookup_url_ = base_url.format(state_, county_, year_, month_)

    # request the website
    driver_.get(lookup_url_)

    _ = WebDriverWait(driver_, 60).until(EC.visibility_of_all_elements_located((By.XPATH,
                                                                                '//tr[@class="ng-star-inserted"]')))

    page_source = driver_.page_source

    soup = BeautifulSoup(page_source, 'lxml')

    table_element = soup.find('table', class_='days')

    head_el = table_element.find('thead')
    body_el = table_element.find('tbody')

    column_names = []
    column_names_sub = []
    data = []

    # month_abbr = "{}".format(month_)

    for column_first_level in head_el.findAll('td', class_="ng-star-inserted"):
        column_names.append(column_first_level.text)

    for column_second_level_td in body_el.find("tr").findAll('td', class_="ng-star-inserted"):
        column_data = []

        table_el = column_second_level_td.find('table')

        if table_el:
            for i, tr_el in enumerate(table_el.findAll('tr', class_='ng-star-inserted')):
                if i == 0:
                    column_names_sub.append(tr_el.text.strip())
                else:
                    column_data.append(tr_el.text.strip())

            data.append(column_data)
        else:
            # print("empty table element for unknown reason")
            pass

    columns = []

    try:
        assert(len(column_names) == len(column_names_sub) == len(data))
    except:
        print("list lengths not equal")
        exit()

    for i in range(len(column_names)):
        left = column_names[i]
        right_names = column_names_sub[i].split()

        columns.extend("_".join([left, right, "wu"]) for right in right_names)

    df_ = pd.DataFrame(columns=columns)

    flat_data = []

    for i, col_list in enumerate(data):
        num_cols = 1
        temp_data = []
        for j, d in enumerate(col_list):
            temp = d.split()
            if len(temp) > 1:
                num_cols = len(temp)

            temp_data.extend(temp)

        flat_data_array = np.array(temp_data).astype(np.float64).reshape(-1, num_cols)

        for c in range(flat_data_array.shape[-1]):
            flat_data.append(flat_data_array[:, c])

    for column, column_data in zip(columns, flat_data):
        df_[column] = pd.Series(column_data).copy()

    word_column = "day"

    for c in df_.columns:
        if re.match(r"Time.+", c):
            word_column = c

    df_["Year_wu"] = year_
    df_["month_wu"] = month_
    df_["state_wu"] = state_
    df_["county_wu"] = county_
    df_.rename(columns={word_column: "day_wu"}, inplace=True)

    return df_


def get_range_data(state_, county_, start_year_, end_year_, start_month_, end_month_, driver_):
    start_day = datetime.datetime(start_year_, start_month_, 1)
    end_day = datetime.datetime(end_year_, end_month_, 1)

    while start_day < end_day:
        print("Scraping for state {}, county {} within range {}".format(state_, county_, start_day.strftime("%Y-%m")))
        y = start_day.year
        m = start_day.month

        try:
            temp_df = get_dayily_weather_from_month_table(state_, county_, y, m, driver_)
        except:
            print("Error happens when scraping for state {}, county {} within range {}".format(state_, county_, start_day.strftime("%Y-%m")))

        start_day = start_day + relativedelta(months=1)
        # print(temp_df.head())
        # print(temp_df.tail())
        # print("------------------------------------")


if __name__ == '__main__':
    start_year = 1980
    end_year = 2022
    start_month = 1
    end_month = 1

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')

    driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=options)

    corn_yield_with_fips = pd.read_csv('../data/corn_yield_with_fips.csv').dropna()
    groups = corn_yield_with_fips.groupby(by=["StateAbbr", "County"]).agg(min)

    state_county_list = list(groups.index)

    for state_county in state_county_list:
        state = state_county[0]
        county = "".join(state_county[1].split())

        get_range_data(state, county, start_year, end_year, start_month, end_month, driver)
        time.sleep(3)
