import requests
import pandas as pd
import pickle
import numpy as np
import datetime
import re
from ediblepickle import checkpoint

# for each county, scrap weather data

'''
    iterate fips code of each county
        select data range
                select data items
                    determine which station to use
                        save the data of each county to a dataframe
'''


# # read corn yield with fips
# corn_yield_with_fips = pd.read_csv('../data/corn_yield_with_fips.csv').dropna()
#
# fips_codes = pd.Series(corn_yield_with_fips['CountyFIPS'].unique().copy())
# fips_codes = pd.to_numeric(fips_codes, downcast='signed').astype(str)
#
# temp_fips = fips_codes[0]

# @checkpoint(
#     key=lambda args, kwargs: '_'.join(
#         [args[0], *args[1]['sids'], args[3].strftime("%Y-%m-%d"), args[4].strftime("%Y-%m-%d")]),
#     # work_dir='C:/Users/45463/Desktop/tdi_corn_yield/tdi_CornYield/scraper/cache/station_weather', refresh=False)
#     work_dir = './cache/station_weather', refresh = False)
def get_station_weather(county_fips_, station_meta_, url_data_, start_day_, end_day_, elem_list_):
    sid_ = station_meta_['sids'][0]
    sid_list = station_meta_['sids']
    station_name = station_meta_['name']
    station_state = station_meta_['state']
    station_longitude = station_meta_['ll'][0]
    station_latitude = station_meta_['ll'][1]

    if 'elev' in station_meta_:
        station_elev = station_meta_['elev']
    else:
        station_elev = np.nan

    start_day_str_ = start_day_.strftime("%Y-%m-%d")
    end_day_str_ = end_day_.strftime("%Y-%m-%d")

    params_data = dict(sid=sid_,
                       sdate=start_day_str_,
                       edate=end_day_str_,
                       elems=','.join(elem_list_),
                       meta="name"
                       )

    response = requests.get(url_data_, params=params_data)
    # if res.status_code == requests.codes.ok:
    meta = response.json()['meta']
    data = response.json()['data']
    columns = ['date']
    columns.extend(elem_list_)

    df = pd.DataFrame(data, columns=columns)

    df = df.applymap(lambda s: re.sub(r"[A-Za-z]*", "", s) if re.match(r"[A-Za-z]*\d+.\d+[A-Za-z]*", s) else s)
    df = df.applymap(lambda s: "M" if re.match(r"[A-Za-z]+", s) else s)

    missing_mask = df.applymap(lambda x: x == 'M')
    tracing_mask = df.applymap(lambda x: x == 'T')

    # df = df.map(lambda s: re.sub(r"[A-Za-z]*", "", s) if re.match(r"[A-Za-z]*\d+.\d+[A-Za-z]*", s) else s)
    # df = df.map(lambda s: "M" if re.match(r"[A-Za-z]+", s) else s)
    #
    # missing_mask = df.map(lambda x: x == 'M')
    # tracing_mask = df.map(lambda x: x == 'T')

    df[missing_mask] = np.nan
    df[tracing_mask] = 0.001

    df['sids'] = '_'.join(sid_list)
    df['name'] = station_name
    df['state'] = station_state
    df['long'] = station_longitude
    df['lati'] = station_latitude
    df['elev'] = station_elev
    # date_list = pd.date_range(start=start_day_, end=end_day_).to_list()
    # df.insert(0, 'date', date_list)  # caused error because date already exists
    # df = df.drop(['date'], axis=1).astype(np.float64)
    return df


# @checkpoint(
#     key=lambda args, kwargs: '_'.join([args[0], args[2].strftime("%Y-%m-%d"), args[3].strftime("%Y-%m-%d")]),
#     # work_dir='C:/Users/45463/Desktop/tdi_corn_yield/tdi_CornYield/scraper/cache/county_weather', refresh=False)
#     work_dir='./cache/county_weather', refresh=False)
def get_county_weather(county_fips, url_meta_, start_day_, end_day_, elems_list_, meta_list_):
    start_day_str_ = start_day_.strftime("%Y-%m-%d")
    end_day_str_ = end_day_.strftime("%Y-%m-%d")

    params_meta = dict(county=county_fips, output='json',
                       elems=','.join(elems_list_),
                       meta=','.join(meta_list_),
                       sdate=start_day_str_,
                       edate=end_day_str_)

    res = requests.get(url_meta_, params=params_meta)

    station_metadata = res.json()['meta']

    url_data = 'http://data.rcc-acis.org/StnData'

    dfs_county = []

    county_longitude_list = []
    county_latitude_list = []
    county_elev_list = []

    for sidex, station_meta in enumerate(station_metadata):
        # sid_list = station_meta['sids']
        # station_name = station_meta['name']
        # station_state = station_meta['state']
        station_longitude = station_meta['ll'][0]
        station_latitude = station_meta['ll'][1]

        if 'elev' in station_meta:
            station_elev = station_meta['elev']
        else:
            station_elev = np.nan

        county_longitude_list.append(station_longitude)
        county_latitude_list.append(station_latitude)
        county_elev_list.append(station_elev)

        df_station = get_station_weather(county_fips, station_meta, url_data, start_day_, end_day_, elems_list_)
        dfs_county.append(df_station[elems_list_].astype(np.float64))

        # limit 50 stations per county
        if sidex >= 5:
            break

    mean_county_longitude = pd.Series(county_longitude_list).astype(np.float64).mean()
    mean_county_latitude = pd.Series(county_latitude_list).astype(np.float64).mean()
    mean_county_elev = pd.Series(county_elev_list).astype(np.float64).mean()

    mean_df_county = pd.concat(dfs_county).groupby(level=0).mean()

    mean_df_county['longitude'] = mean_county_longitude
    mean_df_county['latitude'] = mean_county_latitude
    mean_df_county['elev'] = mean_county_elev
    mean_df_county['county_fips'] = county_fips

    date_list = pd.date_range(start=start_day_, end=end_day_).to_list()

    mean_df_county.insert(0, 'date', date_list)
    # print(mean_df_county)
    return mean_df_county


def read_county_weather(county_fips):
    url_meta_ = "http://data.rcc-acis.org/StnMeta"

    elems_list_ = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd', 'cdd', 'hdd', 'gdd']
    meta_list_ = ['sids', 'name', 'state', 'll', 'elev', 'valid_daterange']
    start_day_ = datetime.date(1980, 1, 1)
    end_day_ = datetime.date(2022, 1, 1)

    df_county = get_county_weather(county_fips, url_meta_, start_day_, end_day_, elems_list_, meta_list_)
    return df_county


def read_county_weather_until_yesterday(county_fips):
    url_meta_ = "http://data.rcc-acis.org/StnMeta"

    elems_list_ = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd', 'cdd', 'hdd', 'gdd']
    meta_list_ = ['sids', 'name', 'state', 'll', 'elev', 'valid_daterange']

    now = datetime.datetime.now()

    end_year = now.year
    end_month = now.month
    end_day = now.day

    if end_month < 11:
        start_year = end_year - 1
    else:
        start_year = end_year

    start_month = 11
    start_day = 1

    start_date_ = datetime.datetime(start_year, start_month, start_day)
    end_date_ = datetime.datetime(end_year, end_month, end_day) - datetime.timedelta(1)

    print(start_date_, end_date_)

    df_county = get_county_weather(county_fips, url_meta_, start_date_, end_date_, elems_list_, meta_list_)
    return df_county


def read_county_weather_custom_year(county_fips, growing_year):
    # here a year is a growing year, which means that if 2012 is given,
    # the weather from Nov. 2011 to Oct. 2012 would be extracted, and the corn yield is extracted from Year 2012
    url_meta_ = "http://data.rcc-acis.org/StnMeta"

    elems_list_ = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd', 'cdd', 'hdd', 'gdd']
    meta_list_ = ['sids', 'name', 'state', 'll', 'elev', 'valid_daterange']

    end_year = growing_year
    end_month = 10
    end_day = 31

    start_year = end_year - 1
    start_month = 11
    start_day = 1

    start_date_ = datetime.datetime(start_year, start_month, start_day)
    end_date_ = datetime.datetime(end_year, end_month, end_day)

    df_county = get_county_weather(county_fips, url_meta_, start_date_, end_date_, elems_list_, meta_list_)
    return df_county


def read_county_weather_last_year(county_fips):
    url_meta_ = "http://data.rcc-acis.org/StnMeta"

    elems_list_ = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd', 'cdd', 'hdd', 'gdd']
    meta_list_ = ['sids', 'name', 'state', 'll', 'elev', 'valid_daterange']

    now = datetime.datetime.now()

    end_year = now.year - 2
    end_month = 10
    end_day = 31

    start_year = end_year - 1
    start_month = 11
    start_day = 1

    start_date_ = datetime.datetime(start_year, start_month, start_day)
    end_date_ = datetime.datetime(end_year, end_month, end_day)

    print(start_date_, end_date_)

    df_county = get_county_weather(county_fips, url_meta_, start_date_, end_date_, elems_list_, meta_list_)
    return df_county


def read_county_weather_until_last_month(county_fips):
    url_meta_ = "http://data.rcc-acis.org/StnMeta"

    elems_list_ = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd', 'cdd', 'hdd', 'gdd']
    meta_list_ = ['sids', 'name', 'state', 'll', 'elev', 'valid_daterange']

    now = datetime.datetime.now()

    end_year = now.year
    end_month = now.month - 1
    end_day = (datetime.datetime(now.year, now.month, 1) - datetime.timedelta(1)).day

    if end_month < 11:
        start_year = end_year - 1
    else:
        start_year = end_year

    start_month = 11
    start_day = 1

    start_date_ = datetime.datetime(start_year, start_month, start_day)
    end_date_ = datetime.datetime(end_year, end_month, end_day)

    print(start_date_, end_date_)

    df_county = get_county_weather(county_fips, url_meta_, start_date_, end_date_, elems_list_, meta_list_)
    return df_county


if __name__ == '__main__':
    url_meta = "http://data.rcc-acis.org/StnMeta"

    elems_list = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd', 'cdd', 'hdd', 'gdd']
    meta_list = ['sids', 'name', 'state', 'll', 'elev', 'valid_daterange']

    start_day = datetime.date(1980, 1, 1)
    end_day = datetime.date(2022, 1, 1)

    corn_yield_with_fips = pd.read_csv('../data/corn_yield_with_fips.csv').dropna()

    fips_codes = pd.Series(corn_yield_with_fips['CountyFIPS'].unique().copy())
    fips_codes = pd.to_numeric(fips_codes, downcast='signed').astype(str)

    print(fips_codes)

    df = pd.DataFrame()

    for i, fips in enumerate(fips_codes):
        print("Current County {}".format(i))
        try:
            temp_df = get_county_weather(fips, url_meta, start_day, end_day, elems_list, meta_list)

            if temp_df.shape[0] != 0 and temp_df.shape[1] != 0:
                df = pd.concat([df, temp_df])
        except:
            print("Error reading {}'th county: {}".format(i, fips))
        #
        # if i > 2:
        #     break

    print(df.head())
    print(df.tail())
    print(df.shape)

    df.to_csv("../data/county_weather.csv", index=False)

    # temp_fips = '17001'
    # print(temp_fips)
    # get_county_weather(temp_fips, url_meta, start_day, end_day, elems_list, meta_list)

    # problematic counties
    # 21201/485  21605/556

