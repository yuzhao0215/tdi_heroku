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
from sklearn.metrics import r2_score

import sys
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


sys.path.insert(0, "../scraper")

from scraper_weather_mrcc import read_county_weather

from lstm_utils import *


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


def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummies], axis=1).drop(columns=[col_name])


def get_daily_df(county_fips_, start_month_=4, end_month_=11):
    df_ = read_county_weather(county_fips_)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()
    df_["is_growing_season"] = df_["month"].apply(lambda x: 0 if end_month_ <= x or x < start_month_ else 1)

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)
    df_.reset_index(inplace=True)

    # add the column of the growing year
    df_["growing_year"] = df_["date"].apply(lambda x: x.year-1 if x.month <= 10 else x.year)

    # add the column of the target year
    df_["target_year"] = df_["growing_year"] + 1

    df_ = df_.drop(columns=["index", "date", "longitude", "latitude", "elev", "county_fips"])  # doesn't drop isgrowingseason column

    # filter corn yield data according to FIPS
    corn_yields_ = pd.read_csv("../data/corn_yield_with_fips.csv", index_col=None)
    corn_yields_['CountyFIPS'] = corn_yields_['CountyFIPS'].apply(lambda x: str(x).strip(".0")).copy()
    corn_yields_ = corn_yields_[corn_yields_['CountyFIPS'] == county_fips_]
    corn_yields_ = corn_yields_.drop(columns=["CountyName", "State"]).copy()

    # merge on growing year and target year
    df_daily_ = pd.merge(df_, corn_yields_, left_on='growing_year', right_on='Year', how="inner")
    df_daily_ = df_daily_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])
    df_daily_ = df_daily_.rename(columns={'Yield': "Actual_Yield"})

    df_daily_ = pd.merge(df_daily_, corn_yields_, left_on='target_year', right_on='Year', how="inner")
    df_daily_ = df_daily_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr', 'Year'])   # drop twice
    df_daily_ = df_daily_.rename(columns={'Yield': "Target_Yield"})

    df_daily_.drop(columns=['growing_year', 'target_year'])

    df_daily_ = onehot_encode_pd(df_daily_, 'month')
    df_daily_ = onehot_encode_pd(df_daily_, 'day')

    return df_daily_


if __name__ == '__main__':
    # read corn yield for county 17001
    county_fips = "17001"

    df_daily = get_daily_df(county_fips)
    indexes = df_daily.index

    # target_column = 'Yield'
    # features = df_daily.columns
    #
    # forecast_lead = 31 + 30 + 31  # aug + sep + oco
    #
    # target = f"{target_column}_lead{forecast_lead}"
    # df_daily[target] = df_daily[target_column].shift(-forecast_lead)
    # df_daily = df_daily.iloc[:-forecast_lead]

    target_column = 'Target_Yield'
    features = df_daily.columns.difference(['Target_Yield'])

    target = f"{target_column}"

    torch.manual_seed(101)

    batch_size = 100
    sequence_length = 1000

    test_start = int(len(df_daily) * 0.75)

    df_train = df_daily.loc[:test_start].copy()

    if test_start - sequence_length + 1 >= 0:
        df_test = df_daily.loc[test_start - sequence_length + 1:].copy()
    else:
        df_test = pd.DataFrame()
        print("Not enough sequence for test dataset, exiting.")
        exit()
    # exp. test_start = 10; when seq = 1, then no need to go backward. when seq = 2, then need to go back additional one.

    print("Test set fraction:", (len(df_test) - sequence_length + 1) / len(df_daily))

    # normalize train and test data sets using mean and std of train data set
    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

        df_train[c].fillna(value=mean, inplace=True)
        df_test[c].fillna(value=mean, inplace=True)

    # print(df_train)

    # training data set
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        is_test=False,
        sequence_length=sequence_length
    )

    # testing data set
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        is_test=True,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = 1e-4
    num_hidden_units = 200

    model = ShallowRegressionLSTM(num_features=len(features), hidden_units=num_hidden_units, num_layers=1)
    model.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(train_loader, model, loss_function)
    test_model(test_loader, model, loss_function)
    print()

    epochs = 50

    for ix_epoch in range(epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()

    # drop the first sequence_len - 1 rows of df_test
    df_test = df_test.iloc[sequence_length - 1:, :]
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean

    original_target = df_test[target] * target_stdev + target_mean
    predict_target = df_test[ystar_col] * target_stdev + target_mean

    mean_error_rate = np.mean(np.absolute((original_target - predict_target) / original_target))
    print("The final predication error rate is: {}%".format(mean_error_rate * 100))
    print("The final r2 score is: {}".format(r2_score(original_target, predict_target)))
    # mean_mse_error = np.mean((original_target - predict_target) ** 2)
    # print("The final predication mse error is: {}\n".format(mean_mse_error))
    #
    # mean_error_rate_using_target_mean = np.mean(np.absolute((original_target - target_mean) / original_target))
    # print("The prediction error rate using target mean is: {}%".format(mean_error_rate_using_target_mean * 100))
    #
    # mean_mse_error_using_target_mean = np.mean((original_target - target_mean) ** 2)
    # print("The mse error using target mean is: {}\n".format(mean_mse_error_using_target_mean))

    # print(df_out)

    fig = px.line(df_out, labels={'value': "Corn Yield (BU/ACRE)", 'created_at': 'Year'})
    fig.add_vline(x=test_start, line_width=4, line_dash="dash")
    fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
    fig.update_layout(
        template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.show()
