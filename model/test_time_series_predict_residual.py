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

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import scipy.fft

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


def add_growing_season(df_, start_month_=4, end_month_=11):
    df_["is_growing_season"] = df_["date"].apply(lambda x: 0 if end_month_ <= x.month or x.month < start_month_ else 1)
    return df_


def get_county_weather(county_fips_):
    df_ = read_county_weather(county_fips_)
    df_ = calculate_accu_gdd_pcpn(df_)
    df_ = add_growing_season(df_)
    df_["year"] = df_["date"].apply(lambda x: x.year).copy()
    return df_


def fft_plot(df_, column_):
    fft = scipy.fft.fft((df_[column_] - df_[column_].mean()).values)

    plt.plot(np.abs(fft))

    plt.title("FFT of temperature data")
    plt.xlabel('# Cycles in full window of data (~40 years)')

    plt.show()


# read corn yield for county 17001
county_fips = "17001"

df_county_weather = get_county_weather(county_fips)

# filter corn yield data according to FIPS
corn_yields = pd.read_csv("../data/corn_yield_with_fips.csv", index_col=None)
corn_yields['CountyFIPS'] = corn_yields['CountyFIPS'].apply(lambda x: str(x).strip(".0")).copy()
corn_yields = corn_yields[corn_yields['CountyFIPS'] == county_fips]
corn_yields = corn_yields.drop(columns=["CountyName", "State"]).copy()

# TODO there should be a better way to join two dfs
df_county_data = pd.merge(df_county_weather, corn_yields, left_on="year", right_on="Year")
df_county_data = df_county_data.drop(columns=['County', 'CountyFIPS', 'StateName', 'StateAbbr'])
df_county_data.drop(columns=['year', "Year"], inplace=True)

# # set date as index
df_county_data.set_index("date", drop=False, inplace=True)

df_county_data['Julian'] = df_county_data.index.to_julian_date()

linear_model_df = df_county_data.copy()

linear_model = Ridge().fit(X=linear_model_df["Julian"].values.reshape(-1, 1), y=linear_model_df["Yield"])
linear_model_df["linear_model"] = linear_model.predict(linear_model_df["Julian"].values.reshape(-1, 1))
linear_model_df["linear_model_error"] = linear_model_df["Yield"] - linear_model_df["linear_model"]

# print(linear_model_df)
linear_model_df["Yield"].fillna(method='ffill', inplace=True)
linear_model_df["linear_model_error"].fillna(method='ffill', inplace=True)

PREDICTION_LAG = 30 #365.25 * 4  # days

CUT_YEAR = 2012

print(linear_model_df)

linear_model_df['Future_Yield'] = linear_model_df["Yield"].shift(-PREDICTION_LAG)

# Train/Test
train = linear_model_df[linear_model_df.index.year < CUT_YEAR]
test = linear_model_df[linear_model_df.index.year >= CUT_YEAR]


# Train the regression
def frame_to_feats(frame):
    feats = pd.DataFrame()

    # test
    feats['LME'] = frame['linear_model_error']
    feats['LME_1'] = frame['linear_model_error'].shift(1)
    feats['dLME_avg'] = pd.Series.rolling(frame['Yield'].diff(), window=PREDICTION_LAG).mean()
    feats['vol_avg'] = pd.Series.ewm(frame['Yield'], span=PREDICTION_LAG).var(bias=False)

    feats['Future_LME'] = frame['linear_model_error'].shift(-PREDICTION_LAG)
    return feats


feats = frame_to_feats(train).dropna(how='any')
X_train = feats.drop('Future_LME', axis=1).values
y_train = feats['Future_LME'].values
regress = LinearRegression().fit(X_train, y_train)

feats = frame_to_feats(test).dropna(how='any')
X_test = feats.drop('Future_LME', axis=1).values
y_test = feats['Future_LME'].values
feats['Predicted_future_LME'] = regress.predict(X_test)

test = feats.join(test, rsuffix='_r').dropna(how='any')
test['Simple_Model'] = test['linear_model'] + test['Predicted_future_LME']

print(test)

# Report
test_me = test[['Future_Yield', 'Simple_Model']].dropna(how='any').rename(columns={'Simple_Model': 'Model'})

from sklearn.metrics import r2_score

print(r2_score(test_me['Future_Yield'], test_me['Model']))
exit()

# ---------------------------------Plotting-----------------------------------#

plot_df = linear_model_df

# create dash app
app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.P('Please select year:', style={'font-size': 20}), width={'size': 6}),
        dbc.Col(dcc.Dropdown(
            id='column-dropdown',
            options=plot_df.columns,
            value="avgt",
        ), width={'size': 6, 'offset': 0})
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-graph'), width={'size': 12, 'offset': 0})
    ])
]
)


@app.callback(
    Output('time-series-graph', 'figure'),
    Input('column-dropdown', 'value'))
def update_map_graph(column):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df[column], mode="lines", name=column), secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df["Yield"], mode="lines", name="Yield"), secondary_y=True
    )

    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      width=1500, height=500,
                      autosize=True,
                      paper_bgcolor="LightSteelBlue")

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
