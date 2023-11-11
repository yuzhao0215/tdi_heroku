import numpy as np
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = None
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service


# Use .format(YYYY, M, D)
lookup_URL = 'https://www.wunderground.com/hourly/us/ny/new-york-city/date/{}-{}-{}.html'
start_date = date.today() + pd.Timedelta(days=1)
end_date = date.today() + pd.Timedelta(days=4)
df_prep = pd.DataFrame()

options = webdriver.ChromeOptions();
options.add_argument('headless');  # to run chrome in the backbroung

driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=options)

while start_date != end_date:
    print('gathering data from: ', start_date)
    formatted_lookup_URL = lookup_URL.format(start_date.year,
                                             start_date.month,
                                             start_date.day)

    driver.get(formatted_lookup_URL)
    rows = WebDriverWait(driver, 60).\
        until(EC.visibility_of_all_elements_located((By.XPATH,
                                                     '//td[@class="mat-cell cdk-cell cdk-column-liquidPrecipitation mat-column-liquidPrecipitation ng-star-inserted"]')))
    for row in rows:
        prep = row.find_element(by=By.XPATH, value='.//span[@class="wu-value wu-value-to"]').text
        df_prep = pd.concat([df_prep, pd.DataFrame({"Day": [str(start_date.day)], 'Precipitation': [prep]})], ignore_index=True)

    start_date += timedelta(days=1)
print(df_prep)