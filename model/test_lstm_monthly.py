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


def get_monthly_df(county_fips_, start_month_=4, end_month_=11):
    df_ = read_county_weather(county_fips_)

    # extract year and month and daily data. TODO: should be a easier way for pandas to group data by datetime column
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    df_["day"] = df_["date"].apply(lambda x: x.day).copy()
    df_["is_growing_season"] = df_["month"].apply(lambda x: 0 if end_month_ <= x or x < start_month_ else 1)

    # calculate accumulated gdd and pcpn from April 1st to next year's Third 31st
    df_ = calculate_accu_gdd_pcpn(df_)
    df_ = df_.drop(columns=["date", "day", "longitude", "latitude", "elev", "county_fips"])  # doesn't drop isgrowingseason column
    # df_ = df_.drop(columns=["date", "day", "longitude", "latitude", "elev", "county_fips", "is_growing_season"])

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
    df_monthly_ = df_.groupby(['year', 'month']).agg(agg_dict_)

    df_monthly_.columns = df_monthly_.columns.map(lambda x: '_'.join(map(str, x)))
    df_monthly_.reset_index(inplace=True)

    # filter corn yield data according to FIPS
    corn_yields_ = pd.read_csv("../data/corn_yield_with_fips.csv", index_col=None)
    corn_yields_['CountyFIPS'] = corn_yields_['CountyFIPS'].apply(lambda x: str(x).strip(".0")).copy()
    corn_yields_ = corn_yields_[corn_yields_['CountyFIPS'] == county_fips_]
    corn_yields_ = corn_yields_.drop(columns=["CountyName", "State"]).copy()

    df_monthly_ = pd.merge(df_monthly_, corn_yields_, left_on='year', right_on='Year', how="inner")
    df_monthly_ = df_monthly_.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr'])

    return df_monthly_


def onehot_encode_pd(df, col_name):
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummies], axis=1).drop(columns=[col_name])


if __name__ == '__main__':
    # read corn yield for county 17001
    county_fips = "17001"

    df_monthly = get_monthly_df(county_fips)
    # df_monthly = df_monthly.drop(columns=['year', 'month'])   # drop year and month columns
    df_monthly = onehot_encode_pd(df_monthly, 'month')

    indexes = df_monthly.index

    target_column = 'Yield'
    features = df_monthly.columns

    forecast_lead = 3

    target = f"{target_column}_lead{forecast_lead}"
    df_monthly[target] = df_monthly[target_column].shift(-forecast_lead)
    df_monthly = df_monthly.iloc[:-forecast_lead]

    torch.manual_seed(101)

    batch_size = 10
    sequence_length = 60

    test_start = int(len(df_monthly) * 0.75)

    df_train = df_monthly.loc[:test_start].copy()

    if test_start - sequence_length + 1 >= 0:
        df_test = df_monthly.loc[test_start - sequence_length + 1:].copy()
    else:
        df_test = pd.DataFrame()
        print("Not enough sequence for test dataset, exiting.")
        exit()
    # exp. test_start = 10; when seq = 1, then no need to go backward. when seq = 2, then need to go back additional one.

    print("Test set fraction:", (len(df_test) - sequence_length + 1) / len(df_monthly))

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

    print(df_train)

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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = 1e-5
    num_hidden_units = 100

    model = ShallowRegressionLSTM(num_features=len(features), hidden_units=num_hidden_units, num_layers=1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    epochs = 200

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
