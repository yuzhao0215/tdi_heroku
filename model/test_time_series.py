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

linear_model_df = df_county_data

linear_model = Ridge().fit(X=linear_model_df["Julian"].values.reshape(-1, 1), y=linear_model_df["Yield"])
linear_model_df["linear_model"] = linear_model.predict(linear_model_df["Julian"].values.reshape(-1, 1))
linear_model_df["linear_model_error"] = linear_model_df["Yield"] - linear_model_df["linear_model"]

# print(linear_model_df)
linear_model_df["avgt"].fillna(method='ffill', inplace=True)
linear_model_df["linear_model_error"].fillna(method='ffill', inplace=True)

linear_model_df['sin(year)'] = np.sin(linear_model_df['Julian'] / 365.25 * 2 * np.pi)
linear_model_df['cos(year)'] = np.cos(linear_model_df['Julian'] / 365.25 * 2 * np.pi)
linear_model_df['sin(6mo)'] = np.sin(linear_model_df['Julian'] / (365.25 / 2) * 2 * np.pi)
linear_model_df['cos(6mo)'] = np.cos(linear_model_df['Julian'] / (365.25 / 2) * 2 * np.pi)


linear_model_df['Goal'] = linear_model_df["avgt"].shift(-(31 + 31 + 30))  # Aug + Sep + Oct

cut_year = 2012

train = linear_model_df[linear_model_df.index.year < cut_year].dropna(how='any')
test = linear_model_df[linear_model_df.index.year >= cut_year].dropna(how='any')

regress = LinearRegression().fit(
    X=train[['avgt', 'sin(year)', 'cos(year)', 'sin(6mo)', 'cos(6mo)']],
    y=train['Goal']
)

test['Predicted_Value'] = regress.predict(X=test[['avgt', 'sin(year)', 'cos(year)', 'sin(6mo)', 'cos(6mo)']])

print(regress.score(test[['avgt', 'sin(year)', 'cos(year)', 'sin(6mo)', 'cos(6mo)']], test['Goal']))

linear_model_df['future_avgt_predicted'] = regress.predict(X=linear_model_df[['avgt', 'sin(year)', 'cos(year)', 'sin(6mo)', 'cos(6mo)']])
linear_model_df['future_avgt_predicted_error'] = linear_model_df['Goal'] - regress.predict(X=linear_model_df[['avgt', 'sin(year)', 'cos(year)', 'sin(6mo)', 'cos(6mo)']])

print(linear_model_df)

# exit()

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
