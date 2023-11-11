# at the top of the file, before other imports
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from pandas.plotting import autocorrelation_plot


import sys
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


sys.path.insert(0, "../scraper")

from scraper_weather_mrcc import read_county_weather

# from lstm_utils import *


pio.templates.default = "plotly_white"              # set plotly style
plot_template = dict(
    layout=go.Layout({
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
)


def calculate_accu_gdd_pcpn(dataframe_):
    selector = ColumnTransformer([('imp', SimpleImputer(strategy='median'), ['gdd', 'pcpn'])])
    filtered_array = selector.fit_transform(dataframe_)

    dataframe_["gdd"] = pd.Series(filtered_array[:, 0])
    dataframe_["pcpn"] = pd.Series(filtered_array[:, 1])

    accu_gdd = []
    accu_pcpn = []

    temp_accu_gdd = 0
    temp_accu_pcpn = 0

    for i in range(len(dataframe_)):
        month = dataframe_.loc[i, 'date'].month
        day = dataframe_.loc[i, 'date'].day
        gdd = dataframe_.loc[i, 'gdd']
        pcpn = dataframe_.loc[i, 'pcpn']

        if month == 4 and day == 1:
            temp_accu_gdd = gdd
            temp_accu_pcpn = pcpn
        else:
            temp_accu_gdd += gdd
            temp_accu_pcpn += pcpn

        if np.isnan(temp_accu_pcpn):
            print(pcpn)

        # print("month: {}, day: {}, gdd: {}, accu_gdd: {}, accu_pcpn: {}".format(month, day, gdd, temp_accu_gdd, temp_accu_pcpn))

        accu_gdd.append(temp_accu_gdd)
        accu_pcpn.append(temp_accu_pcpn)

    dataframe_['accu_gdd'] = pd.Series(accu_gdd)
    dataframe_['accu_pcpn'] = pd.Series(accu_pcpn)
    return dataframe_


def get_time_series(df_, lag_=5):
    columns = df_.columns.difference(['year', 'month'])

    for i in range(1, lag_):
        for col in columns:
            df_[f"{col}_lag_{i}"] = df_[col].shift(i)

    return df_.dropna(how='any')


def get_monthly_df(county_fips_, start_month_=4, end_month_=11):
    df_ = read_county_weather(county_fips_)

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)
    df_ = df_.drop(columns=["longitude", "latitude", "elev", "county_fips"])  # doesn't drop isgrowingseason column
    df_.set_index('date', drop=True, inplace=True)

    agg_columns = df_.columns
    agg_attributes_ = [['min', 'max', 'mean']] * len(agg_columns)
    agg_dict_ = dict(zip(agg_columns, agg_attributes_))

    df_monthly_ = df_.groupby([df_.index.year, df_.index.month]).agg(agg_dict_)
    df_monthly_.columns = df_monthly_.columns.map(lambda x: '_'.join(map(str, x)))
    df_monthly_.index.set_names(['year', 'month'], inplace=True)
    df_monthly_.reset_index(inplace=True)

    df_monthly_ = get_time_series(df_monthly_, lag_=1)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_monthly_["is_growing_season"] = df_monthly_["month"].apply(lambda x: 0 if end_month_ <= x or x < start_month_ else 1).copy()

    # add the column of the growing year
    df_monthly_['growing_year'] = df_monthly_.apply(lambda row: row['year'] - 1 if row['month'] <= 10 else row['year'], axis=1).copy()

    # add the column of the target year
    df_monthly_["target_year"] = df_monthly_["growing_year"] + 1
    # df_monthly_["growing_year_lag_1"] = df_monthly_["growing_year"] - 1
    # df_monthly_["growing_year_lag_2"] = df_monthly_["growing_year"] - 2
    # df_monthly_["growing_year_lag_3"] = df_monthly_["growing_year"] - 3

    # filter corn yield data according to FIPS
    corn_yields_ = pd.read_csv("../data/corn_yield_with_fips.csv", index_col=None)
    corn_yields_['CountyFIPS'] = corn_yields_['CountyFIPS'].apply(lambda x: str(x).strip(".0")).copy()
    corn_yields_ = corn_yields_[corn_yields_['CountyFIPS'] == county_fips_]
    corn_yields_ = corn_yields_.drop(columns=["CountyName", "State"]).copy()
    #
    # merge on growing year and target year
    df_monthly_ = pd.merge(df_monthly_, corn_yields_, left_on='growing_year', right_on='Year', how="inner")
    df_monthly_ = df_monthly_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])
    df_monthly_ = df_monthly_.rename(columns={'Yield': "Actual_Yield"})
    #
    df_monthly_ = pd.merge(df_monthly_, corn_yields_, left_on='target_year', right_on='Year', how="inner")
    df_monthly_ = df_monthly_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])   # drop twice
    df_monthly_ = df_monthly_.rename(columns={'Yield': "Target_Yield"})

    # df_monthly_ = pd.merge(df_monthly_, corn_yields_, left_on='growing_year_lag_1', right_on='Year', how="inner")
    # df_monthly_ = df_monthly_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])   # drop twice
    # df_monthly_ = df_monthly_.rename(columns={'Yield': "Actual_Yield_lag_1"})
    #
    # df_monthly_ = pd.merge(df_monthly_, corn_yields_, left_on='growing_year_lag_2', right_on='Year', how="inner")
    # df_monthly_ = df_monthly_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])   # drop twice
    # df_monthly_ = df_monthly_.rename(columns={'Yield': "Actual_Yield_lag_2"})
    #
    # df_monthly_ = pd.merge(df_monthly_, corn_yields_, left_on='growing_year_lag_3', right_on='Year', how="inner")
    # df_monthly_ = df_monthly_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])   # drop twice
    # df_monthly_ = df_monthly_.rename(columns={'Yield': "Actual_Yield_lag_3"})

    df_monthly_ = df_monthly_.drop(columns=['growing_year', 'target_year'])
    # df_monthly_ = df_monthly_.drop(columns=['growing_year', 'target_year', 'growing_year_lag_1', 'growing_year_lag_2', 'growing_year_lag_3'])

    return df_monthly_


if __name__ == '__main__':
    # read corn yield for county 17001
    county_fips = "17001"

    df_monthly = get_monthly_df(county_fips)
    X = df_monthly.drop(columns=['Target_Yield'])
    y = df_monthly['Target_Yield']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    imputer = SimpleImputer(strategy='median')
    onehot = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    scaler = StandardScaler()
    ridge = Ridge()

    one_hot_transformer = ColumnTransformer(transformers=[('one', onehot, ['month'])],
                                            remainder='passthrough')

    pipeline_test = Pipeline([('onehot', one_hot_transformer), ('imp', imputer), ('scaler', scaler)])
    test = pipeline_test.fit_transform(X_train)

    pipeline = Pipeline([('onehot', one_hot_transformer), ('imp', imputer), ('scaler', scaler), ('ridge', ridge)])

    params_ = {
        'ridge__alpha': [0.01, 1, 10, 100, 500, 1000],
    }

    final_reg = GridSearchCV(pipeline, params_, cv=3, n_jobs=-1).fit(X_train, y_train)

    print(final_reg.best_params_)

    print(final_reg.score(X_train, y_train))
    print(final_reg.score(X_test, y_test))
