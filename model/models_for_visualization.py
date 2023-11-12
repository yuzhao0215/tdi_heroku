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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import sys
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

# sys.path.insert(0, "../scraper")

from scraper.scraper_weather_mrcc import read_county_weather
from scraper.scraper_weather_mrcc import read_county_weather_until_last_month

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
    agg_columns_ = df_.columns.difference(['year', 'month'])

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
    corn_yields_ = corn_yields_.drop(columns=["County", "State"]).copy()

    # select "Year" column as index column, so that unstacked_df and corn_yields can be joined by their indexes
    corn_yields_.set_index("Year", inplace=True)

    # find corn yields for each year
    unstacked_df_ = pd.merge(unstacked_df_, corn_yields_, left_index=True, right_index=True)
    unstacked_df_ = unstacked_df_.drop(columns=['CountyFIPS', 'StateName'])

    # convert "Year" from index to an ordinary column, which will be used in regression
    unstacked_df_.reset_index(inplace=True)
    unstacked_df_ = unstacked_df_.rename(columns={'index': 'Year'})

    # get linear regression error
    linear_reg = LinearRegression()

    linear_X_train = unstacked_df_['Year'].values.reshape(-1, 1)
    linear_X_test = unstacked_df_['Yield']
    linear_reg.fit(linear_X_train, linear_X_test)
    linear_prediction = linear_reg.predict(linear_X_train)

    print("mape for linear model is: {}".format(mean_absolute_percentage_error(unstacked_df_['Yield'], linear_prediction)))

    unstacked_df_["linear_model_prediction"] = linear_prediction
    unstacked_df_["linear_model_error"] = unstacked_df_['Yield'] - linear_prediction

    return unstacked_df_, linear_reg


def split(unstacked_df_, drop_columns_, target_column_):
    X_ = unstacked_df_.drop(columns=drop_columns_).copy()
    y_ = unstacked_df_[target_column_]

    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        X_, y_, test_size=0.33, random_state=42)

    data_list_ = [X_train_, X_test_, y_train_, y_test_, X_, y_]
    return data_list_


def regression(df_yearly_, data_list_, show=False, recording_df=None, current_actual_month_=-1, fips_='17001'):
    imputer_ = SimpleImputer(strategy='median')
    scaler_ = StandardScaler()
    ridge_rg_ = Ridge()
    dt_rg_ = DecisionTreeRegressor()
    rf_rg_ = RandomForestRegressor()
    knn_rg_ = KNeighborsRegressor()

    # pipeline_ = Pipeline([
    #     ('imp', imputer_),
    #     ('scaler', scaler_),
    #     ('ridge', ridge_rg_)
    # ])
    #
    # params_ = {
    #     'ridge__alpha': [0.01, 0.1, 1, 10, 100, 200, 500, 1000],
    # }

    pipeline_ = Pipeline([
        ('imp', imputer_),
        ('scaler', scaler_),
        # ('knn', knn_rg_)
        ('rf', rf_rg_)
        # ('dt', dt_rg_)
    ])

    params_ = {
        # 'dt__max_depth': [1, 2, 3],
        # 'dt__min_samples_leaf': [1, 2, 3, 4, 5]
        'rf__max_depth': [1, 3, 5, 7, 9, 11],
        # 'rf__min_samples_leaf': [1, 2, 3]
        # 'knn__n_neighbors': [3, 5, 7, 9]
    }

    X_train_, X_test_, y_train_, y_test_, X_, y_ = data_list_[0], data_list_[1], data_list_[2], data_list_[3], \
                                                   data_list_[4], data_list_[5],

    if recording_df is not None:
        # check if Yields and linear_model_prediction are in train data
        if 'Yield' in X_train_.columns:
            train_yields = X_train_['Yield']
            X_train_ = X_train_.drop(columns=['Yield'])
            test_yields = X_test_['Yield']
            X_test_ = X_test_.drop(columns=['Yield'])

        if 'linear_model_prediction' in X_train_.columns:
            train_linear_model_prediction = X_train_['linear_model_prediction']
            X_train_ = X_train_.drop(columns=['linear_model_prediction'])
            test_linear_model_prediction = X_test_['linear_model_prediction']
            X_test_ = X_test_.drop(columns=['linear_model_prediction'])

    # fitting
    final_reg_ = GridSearchCV(pipeline_, params_, cv=3, n_jobs=-1)
    final_reg_.fit(X_train_, y_train_)

    print("best params: ", final_reg_.best_params_)
    print("Training score: {}".format(final_reg_.score(X_train_, y_train_)))
    print("Test score: {}".format(final_reg_.score(X_test_, y_test_)))

    # yields = df_yearly_["Yield"].values
    # linear_model_prediction = df_yearly_["linear_model_prediction"]
    # predicted_error = final_reg_.predict(X_)
    #
    # print("Actual Yield score after adding linear prediction: ",
    #       r2_score(yields, linear_model_prediction + predicted_error))

    if recording_df is not None:
        train_prediction = final_reg_.predict(X_train_) + train_linear_model_prediction
        test_prediction = final_reg_.predict(X_test_) + test_linear_model_prediction

        train_r2 = r2_score(train_yields, train_prediction)
        test_r2 = r2_score(test_yields, test_prediction)

        train_mape = mean_absolute_percentage_error(train_yields, train_prediction)
        test_mape = mean_absolute_percentage_error(test_yields, test_prediction)

        dic = dict(current_actual_month=current_actual_month_,
                   train_r2=train_r2,
                   test_r2=test_r2,
                   train_mape=train_mape,
                   test_mape=test_mape,
                   CountyFIPS=fips_
                   )

        # recording_df = pd.concat([recording_df, dic], axis=0, ignore_index=True)

        recording_df.loc[len(recording_df)] = dic

    if recording_df is not None:
        return final_reg_, recording_df
    else:
        return final_reg_


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
    agg_columns_ = df_.columns.difference(['year', 'month'])

    agg_attributes_ = [['min', 'max', 'mean']] * len(agg_columns_)
    agg_dict_ = dict(zip(agg_columns_, agg_attributes_))

    groups_ = df_.groupby(['year', 'month']).agg(agg_dict_)

    unstacked_df_ = groups_.unstack()
    unstacked_df_.columns = unstacked_df_.columns.map(lambda x: '_'.join(map(str, x)))
    unstacked_df_.reset_index(inplace=True)
    unstacked_df_ = unstacked_df_.rename(columns={'year': 'Year'})

    return unstacked_df_


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
    df_county_fips = pd.read_csv('../data/county_fips.csv', index_col=None)
    df_county_fips['CountyFIPS'] = df_county_fips['CountyFIPS'].apply(lambda x: str(x).strip(".0"))

    # drop_columns = ['Yield', 'linear_model_prediction', 'linear_model_error']

    # for county_fips in df_county_fips['CountyFIPS']:
    #     try:
    #         for current_month in range(1, 13):
    #             print(f"month {current_month}:")
    #             df_yearly, linear_model = get_yearly_data(county_fips, current_month)  # the last month is excluded
    #             data_list = split(df_yearly, drop_columns, 'linear_model_error')
    #             model = regression(df_yearly, data_list)
    #
    #             dump(linear_model, f'./cache/linear_models/{county_fips}_linear.joblib')
    #             dump(model, f'./cache/linear_error_models/{county_fips}_{current_month}.joblib')
    #     except:
    #         print(f"error in training county {county_fips}")

    # averaging accuracy for twenty counties
    num_counties = 100
    sample_counties = df_county_fips['CountyFIPS'].sample(n=num_counties, random_state=42)
    # sample_counties = pd.read_csv('../data/top_100_corn_yield_fips.csv')['CountyFIPS'].apply(lambda x: str(x).strip('.0'))[:num_counties]

    accuracy_df = pd.DataFrame(columns=["current_actual_month", "CountyFIPS", "train_r2", "test_r2", "train_mape", "test_mape"])

    drop_columns = ['linear_model_error']

    for county_fips in sample_counties:
        print(f"{county_fips}")
        for current_actual_month in range(1, 13):
            print(f"month {current_actual_month}:")
            try:
                df_yearly, linear_model = get_yearly_data(county_fips, current_actual_month)  # the last month is excluded
                data_list = split(df_yearly, drop_columns, 'linear_model_error')
                model, accuracy_df = regression(df_yearly, data_list,
                                                recording_df=accuracy_df,
                                                current_actual_month_=current_actual_month,
                                                fips_=county_fips)
            except:
                print(f"error in {county_fips}, month: {current_actual_month}")

    accuracy_df.to_csv("../data/accuracy_df.csv", index=False)

