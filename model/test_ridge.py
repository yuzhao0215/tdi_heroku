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


def get_ridge(county_fips_, start_month_=4, end_month_=11):
    df_ = read_county_weather(county_fips_)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()

    # add a column indicating if current day is within the corn growing season, which is between April and October
    df_["is_growing_season"] = df_["month"].apply(lambda x: 0 if end_month_ <= x or x < start_month_ else 1)

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)

    # filter data out that are not in growing season
    df_ = df_[df_["is_growing_season"] == 1].copy()

    # drop columns that will not be used for regression
    df_ = df_.drop(columns=["date", "day", "longitude", "latitude", "elev", "county_fips", "is_growing_season"])

    # construct a dictionary for 'groupby' aggregation
    remaining_columns_ = list(df_.columns)
    agg_attributes_ = [['min', 'max', 'mean']] * len(remaining_columns_)
    agg_dict_ = dict(zip(remaining_columns_, agg_attributes_))

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

    imputer_ = SimpleImputer(strategy='mean')
    scaler_ = StandardScaler()
    ridge_rg_ = Ridge()

    pipeline_ = Pipeline([
        ('imp', imputer_),
        ('scaler', scaler_),
        ('ridge', ridge_rg_)
    ])

    params_ = {
        'ridge__alpha': range(1, 101, 1),
        # 'imp__strategy': ['mean', 'median', 'most_frequent']
    }

    X_ = unstacked_df_.drop(columns=['Yield']).copy()
    y_ = unstacked_df_['Yield']

    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        X_, y_, test_size=0.25, random_state=20)

    final_reg_ = GridSearchCV(pipeline_, params_, cv=3, n_jobs=-1, return_train_score=True)
    final_reg_.fit(X_train_, y_train_)

    data_list_ = [X_train_, X_test_, y_train_, y_test_, X_, y_]

    return final_reg_, corn_yields_, data_list_


if __name__ == '__main__':
    # read corn yield for county 17001
    county_fips = "17001"

    final_reg, corn_yields, data_list = get_ridge(county_fips, 4, 8)  # the last month is excluded
    X_train, X_test, y_train, y_test, X, y = data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5]


    print(final_reg.best_params_)
    # print(final_reg.cv_results_['mean_train_score'])
    # print(final_reg.cv_results_['mean_test_score'])

    plt.plot(X['Year'], y, 'o', color='k', label='actual yield')
    # plt.plot(X_test['Year'], y_test, 'o', color='k', label='test data')
    # plt.plot(X_train['Year'], y_train, 'o', color='b', label='training data')

    plt.plot(X['Year'], final_reg.predict(X), 'o-', color='#42a5f5ff', label='linear model prediction')
    plt.xlabel('Year')
    plt.ylabel('Corn Yield (BU/ACRE)')
    plt.legend()

    print("Training score: {}".format(final_reg.score(X_train, y_train)))
    print("Test score: {}".format(final_reg.score(X_test, y_test)))

    print(corn_yields.dtypes)
    plt.show()
