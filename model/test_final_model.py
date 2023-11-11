import numpy as np
import pandas as pd
import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sys

sys.path.insert(0, "../scraper")

from scraper_weather_mrcc import read_county_weather
from scraper_weather_mrcc import read_county_weather_now


from joblib import dump, load


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
        month = dataframe_.loc[i, 'month']
        day = dataframe_.loc[i, 'day']
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


def find_maximum_actual_month(df_):

    df_ = df_.copy()

    df_.fillna(0, inplace=True)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()

    return df_["month"].values[-1]


def preprocess_by_month(df_, temp_grow_month_):
    df_.fillna(0, inplace=True)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)

    df_['year'] = df_.apply(lambda row: row['year'] if row['month'] <= 10 else row['year'] + 1, axis=1).copy()
    df_['month'] = df_['month'].apply(lambda x: (x - 1 + 2) % 12 + 1)

    df_ = df_[df_["month"] <= temp_grow_month_].copy()

    # drop columns that will not be used for regression
    df_ = df_.drop(columns=["date", "day", "longitude", "latitude", "elev", "county_fips"])

    # construct a dictionary for 'groupby' aggregation
    # agg_columns_ = df_.columns.difference(['year', 'month'])
    # agg_columns_ = df_.columns.difference(['year'])
    agg_columns_ = df_.columns.difference(['month'])
    # todo have no idea why extra year columns can improve r2 score from 0.5 - 0.74, but adding extra month columns didn't help
    # agg_columns_ = list(df_.columns)

    agg_attributes_ = [['min', 'max', 'mean']] * len(agg_columns_)
    agg_dict_ = dict(zip(agg_columns_, agg_attributes_))

    # perform groupby on monthly data
    # the dataframe of groups is like:
    #                  maxt                         ...  accu_pcpn
    #                   min         max       mean  ...        min        max       mean
    # year month                                    ...
    # 1980 4      44.666667   87.333333  62.422222  ...   0.000000   1.994000   1.468133
    #      5      63.333333   88.333333  75.731183  ...   1.994000   5.558167   3.657167
    groups_ = df_.groupby(['year', 'month']).agg(agg_dict_)

    # After unstacking, the dataframe is like:
    #       maxt_min_4  maxt_min_5  ...  accu_pcpn_mean_9  accu_pcpn_mean_10
    # year                          ...
    # 1980   44.666667   63.333333  ...         22.347083          24.088522
    unstacked_df_ = groups_.unstack()
    unstacked_df_.columns = unstacked_df_.columns.map(lambda x: '_'.join(map(str, x)))
    unstacked_df_.reset_index(inplace=True)
    unstacked_df_ = unstacked_df_.rename(columns={'year': 'Year'})

    return unstacked_df_


def preprocess(df_, county_fips_):
    df_.fillna(0, inplace=True)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)

    df_['year'] = df_.apply(lambda row: row['year'] if row['month'] <= 10 else row['year'] + 1, axis=1).copy()
    df_['month'] = df_['month'].apply(lambda x: (x - 1 + 2) % 12 + 1)
    current_month_ = df_["month"].values[-1]

    df_ = df_[df_["month"] <= current_month_].copy()

    # drop columns that will not be used for regression
    df_ = df_.drop(columns=["date", "day", "longitude", "latitude", "elev", "county_fips"])

    # construct a dictionary for 'groupby' aggregation
    # agg_columns_ = df_.columns.difference(['year', 'month'])
    # agg_columns_ = df_.columns.difference(['year'])
    agg_columns_ = df_.columns.difference(['month'])
    # todo have no idea why extra year columns can improve r2 score from 0.5 - 0.74, but adding extra month columns didn't help
    # agg_columns_ = list(df_.columns)

    agg_attributes_ = [['min', 'max', 'mean']] * len(agg_columns_)
    agg_dict_ = dict(zip(agg_columns_, agg_attributes_))

    # perform groupby on monthly data
    # the dataframe of groups is like:
    #                  maxt                         ...  accu_pcpn
    #                   min         max       mean  ...        min        max       mean
    # year month                                    ...
    # 1980 4      44.666667   87.333333  62.422222  ...   0.000000   1.994000   1.468133
    #      5      63.333333   88.333333  75.731183  ...   1.994000   5.558167   3.657167
    groups_ = df_.groupby(['year', 'month']).agg(agg_dict_)

    # After unstacking, the dataframe is like:
    #       maxt_min_4  maxt_min_5  ...  accu_pcpn_mean_9  accu_pcpn_mean_10
    # year                          ...
    # 1980   44.666667   63.333333  ...         22.347083          24.088522
    unstacked_df_ = groups_.unstack()
    unstacked_df_.columns = unstacked_df_.columns.map(lambda x: '_'.join(map(str, x)))
    unstacked_df_.reset_index(inplace=True)
    unstacked_df_ = unstacked_df_.rename(columns={'year': 'Year'})

    return unstacked_df_, current_month_


def get_yearly_data(county_fips_, current_month_=11):
    df_ = read_county_weather(county_fips_)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)

    df_['year'] = df_.apply(lambda row: row['year'] if row['month'] <= 10 else row['year'] + 1, axis=1).copy()
    df_['month'] = df_['month'].apply(lambda x: (x - 1 + 2) % 12 + 1)

    # # filter data out that are not in growing season
    df_ = df_[df_["month"] <= (current_month_ + 1) % 12 + 1].copy()

    # drop columns that will not be used for regression
    df_ = df_.drop(columns=["date", "day", "longitude", "latitude", "elev", "county_fips"])

    # construct a dictionary for 'groupby' aggregation
    # agg_columns_ = df_.columns.difference(['year', 'month'])
    # agg_columns_ = df_.columns.difference(['year'])
    agg_columns_ = df_.columns.difference(['month'])
    # todo have no idea why extra year columns can improve r2 score from 0.5 - 0.74, but adding extra month columns didn't help
    # agg_columns_ = list(df_.columns)

    agg_attributes_ = [['min', 'max', 'mean']] * len(agg_columns_)
    agg_dict_ = dict(zip(agg_columns_, agg_attributes_))

    # perform groupby on monthly data
    # the dataframe of groups is like:
    #                  maxt                         ...  accu_pcpn
    #                   min         max       mean  ...        min        max       mean
    # year month                                    ...
    # 1980 4      44.666667   87.333333  62.422222  ...   0.000000   1.994000   1.468133
    #      5      63.333333   88.333333  75.731183  ...   1.994000   5.558167   3.657167
    groups_ = df_.groupby(['year', 'month']).agg(agg_dict_)

    # After unstacking, the dataframe is like:
    #       maxt_min_4  maxt_min_5  ...  accu_pcpn_mean_9  accu_pcpn_mean_10
    # year                          ...
    # 1980   44.666667   63.333333  ...         22.347083          24.088522
    unstacked_df_ = groups_.unstack()
    unstacked_df_.columns = unstacked_df_.columns.map(lambda x: '_'.join(map(str, x)))

    # filter corn yield data according to FIPS
    corn_yields_ = pd.read_csv("../data/corn_yield_with_fips.csv", index_col=None)
    corn_yields_['CountyFIPS'] = corn_yields_['CountyFIPS'].apply(lambda x: str(x).strip(".0")).copy()
    corn_yields_ = corn_yields_[corn_yields_['CountyFIPS'] == county_fips_]
    corn_yields_ = corn_yields_.drop(columns=["CountyName", "State"]).copy()

    # select "Year" column as index column, so that unstacked_df and corn_yields can be joined by their indexes
    corn_yields_.set_index("Year", inplace=True)

    # find corn yields for each year
    unstacked_df_ = pd.merge(unstacked_df_, corn_yields_, left_index=True, right_index=True)
    unstacked_df_ = unstacked_df_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr'])

    # convert "Year" from index to an ordinary column, which will be used in regression
    unstacked_df_.reset_index(inplace=True)
    unstacked_df_ = unstacked_df_.rename(columns={'index': 'Year'})

    return unstacked_df_


def split(unstacked_df_):
    X_ = unstacked_df_.drop(columns=['Yield']).copy()
    y_ = unstacked_df_['Yield']

    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        X_, y_, test_size=0.25, random_state=42)

    data_list_ = [X_train_, X_test_, y_train_, y_test_, X_, y_]
    return data_list_


def regression(data_list_, show=False):
    imputer_ = SimpleImputer(strategy='mean')
    scaler_ = StandardScaler()
    ridge_rg_ = Ridge()

    pipeline_ = Pipeline([
        ('imp', imputer_),
        ('scaler', scaler_),
        ('ridge', ridge_rg_)
    ])

    params_ = {
        'ridge__alpha': [0.01, 0.1, 1, 10, 100, 200, 500, 1000],
    }

    X_train_, X_test_, y_train_, y_test_, X_, y_ = data_list_[0], data_list_[1], data_list_[2], data_list_[3], data_list_[4], data_list_[5],

    final_reg_ = GridSearchCV(pipeline_, params_, cv=3, n_jobs=-1, return_train_score=True)
    final_reg_.fit(X_train_, y_train_)

    print("Training score: {}".format(final_reg_.score(X_train_, y_train_)))
    print("Test score: {}".format(final_reg_.score(X_test_, y_test_)))

    if show:
        plt.plot(X_['Year'], y_, 'o', color='k', label='actual yield')
        # plt.plot(X_test['Year'], y_test, 'o', color='k', label='test data')
        # plt.plot(X_train['Year'], y_train, 'o', color='b', label='training data')

        plt.plot(X_['Year'], final_reg_.predict(X_), 'o-', color='#42a5f5ff', label='linear model prediction')
        plt.xlabel('Year')
        plt.ylabel('Corn Yield (BU/ACRE)')
        plt.legend()

        plt.show()

    return final_reg_


def actual_to_growing_month(am):
    return (am + 1) % 12 + 1


def reverse_growing_month(gm):
    if gm == 1:
        return 11
    elif gm == 2:
        return 12
    else:
        return gm - 2


if __name__ == '__main__':
    # read corn yield for county 17001
    county_fips = "17001"

    # models = dict()
    #
    # for m in range(1, 13):
    #     print(f"month {m}:")
    #     df_yearly = get_yearly_data(county_fips, m)  # the last month is excluded
    #     data_list = split(df_yearly)
    #     model = regression(data_list)
    #
    #     models[m] = model
    #     dump(model, f'./cache/{m}.joblib')

    df = read_county_weather_until_last_month(county_fips)
    df, grow_month = preprocess(df, county_fips)

    actual_month = reverse_growing_month(grow_month)

    model = load(f'./cache/{actual_month}.joblib')

    print(model.predict(df))

