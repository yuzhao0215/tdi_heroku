import requests
import json
import pickle
import pandas as pd
from ediblepickle import checkpoint


key_api = '11710A85-AC22-3A4C-9856-61BF8D6BD047'


def get_possible_para_values(param_):
    params = dict(key=key_api,
                  param=param_
                  )

    res = requests.get('http://quickstats.nass.usda.gov/api/get_param_values',
                       params=params)

    if res.status_code == requests.codes.ok:
        results = res.json()
        print("choices for param: {}".format(param_), results)
    else:
        print("Error request url")

@checkpoint(
    key=lambda args, kwargs: '_'.join([args[0], args[1], 'since', args[2]]),
    work_dir='./cache/corn_yield_of_county', refresh=False)
def get_corn_yield(state_alpha_, county_name_, year_):
    url = 'http://quickstats.nass.usda.gov/api/api_GET'

    params = dict(
        key=key_api,
        source_desc='SURVEY',
        sector_desc='CROPS',
        group_desc='FIELD CROPS',
        commodity_desc='CORN',
        statisticcat_desc='YIELD',
        util_practice_desc='GRAIN',
        unit_desc='BU / ACRE',
        state_alpha=state_alpha_,
        county_name=county_name_,
        year__GE=year_
    )

    res = requests.get(url, params=params)
    columns = ['Year', 'Yield', 'State', 'County']
    df_temp_ = pd.DataFrame(columns=[columns])

    if res.status_code == requests.codes.ok:
        data = res.json()['data']

        for d in data:
            year = d['year']
            value = d['Value']
            row = pd.DataFrame([[year, value, state_alpha_, county_name_]], columns=[columns])
            df_temp_ = pd.concat([df_temp_, row], ignore_index=True)
    else:
        print(res.text)
        print("Error request for state {}, county: {} in year {}".format(state_alpha_, county_name_, year_))

    return df_temp_


def get_yield_all_counties(location_dic, start_year_, end_year_):
    column_names = ['Year', 'Yield', 'State', 'County']

    df_total_ = pd.DataFrame(columns=[column_names])

    for state_alpha, counties in location_dic.items():
        for county in counties:
            df_res = get_corn_yield(state_alpha, county, start_year_)

            if df_res.shape[0] == 0:
                print(df_res.shape)
                print(state_alpha, county)

            if df_res.shape[0] != 0:
                df_total_ = pd.concat([df_total_, df_res], ignore_index=True)

    return df_total_


states_alpha2county = pickle.load(open("county_names_dic.pkl", "rb"))

start_year = 1980
end_year = 2021

df_total = get_yield_all_counties(states_alpha2county, str(start_year), str(end_year))
df_total.to_csv('../data/corn_yield.csv', index=False)
print(df_total)