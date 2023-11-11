import sys

# sys.path.insert(0, "../scraper")
# sys.path.insert(0, "../model")

from scraper.scraper_weather_mrcc import read_county_weather
from scraper.scraper_weather_mrcc import read_county_weather_until_last_month
from scraper.scraper_weather_mrcc import read_county_weather_until_yesterday
from scraper.scraper_weather_mrcc import read_county_weather_last_year
from scraper.scraper_weather_mrcc import read_county_weather_custom_year

from model.models_for_visualization import *
from PIL import Image

import datetime
import numpy as np
import pandas as pd

from urllib.request import urlopen
import json

from sklearn.linear_model import LinearRegression

from joblib import dump, load
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
# from whitenoise import WhiteNoise
import requests
import os
import boto3


client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='us-east-2'
)

bucket_name = os.getenv("S3_BUCKET_NAME")
print("--------------Bucket name is: {}".format(bucket_name))

logo_img_obj = client.get_object(
    Bucket=bucket_name,
    Key='image/amaizeing.png'
)
#
corn_yield_with_fips_obj = client.get_object(
    Bucket=bucket_name,
    Key='csv/corn_yield_with_fips.csv'
)

# open county geojson file for corn yield plotting on map
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# save corn_yield_df after merging

corn_yield_with_fips = pd.read_csv(corn_yield_with_fips_obj['Body'])

corn_yield_with_fips['CountyFIPS'] = corn_yield_with_fips['CountyFIPS'].apply(lambda x: str(x).strip('.0'))  # todo better strip later

# latest year of corn yield data
latest_year = corn_yield_with_fips['Year'].max()
oldest_year = corn_yield_with_fips['Year'].min()

df_county = corn_yield_with_fips.groupby('CountyFIPS').agg(min).drop(columns=['Year', 'Yield'])
df_county.reset_index(inplace=True)
df_county['CountyFIPS'] = df_county['CountyFIPS'].apply(lambda x: str(x).strip('.0'))  # todo better strip later

# df_county.to_csv('../data/county_fips.csv', index=False)

states = df_county['StateName'].unique()

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
# server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

offset = 3

DEFAULT_STATE = 'Illinois'
DEFAULT_COUNTY = 'ADAMS'

# logo_img = Image.open("./static/amaizeing.png")
logo_img = Image.open(logo_img_obj['Body'])

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.Img(src=logo_img, style={'width':'100%'}), width={'size': 6, 'offset': 3}),
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select a year for history distribution:', style={'font-size': 20}),
                width={'size': 3, 'offset': 3}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='year',
            options=np.arange(oldest_year, latest_year + 1),
            value=latest_year,
        ), width={'size': 3, 'offset': 0}, lg=3),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='corn-yield-graph'), width={'size': 6, 'offset': 0}, lg=6),
        dbc.Col(dcc.Graph(id='corn-yield-year'), width={'size': 6, 'offset': 0}, lg=6)
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select a state:', style={'font-size': 20}), width={'size': 3, 'offset': 0}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='state-dropdown',
            options=states,
            value=DEFAULT_STATE
        ), width={'size': 3, 'offset': 0}, lg=3),
        dbc.Col(html.P('Please select a county:', style={'font-size': 20}), width={'size': 3, 'offset': 0}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='county-dropdown',
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    # dbc.Row([
    #
    # ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='daily-weather-now'), width={'size': 6, 'offset': 0}, lg=6),
        dbc.Col(dcc.Graph(id='corn-yield-month'), width={'size': 6, 'offset': 0}, lg=6)
    ]),
    # dbc.Row([
    #     dbc.Col(dcc.Graph(id='daily-weather-lastyear'), width={'size': 6, 'offset': 0}, lg=6),
    #     dbc.Col(dcc.Graph(id='corn-yield-month-lastyear'), width={'size': 6, 'offset': 0}, lg=6)
    # ]),
    dbc.Row([
        dbc.Col(html.P('Please select a history year:', style={'font-size': 20}), width={'size': 3, 'offset': 0}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='history-year-dropdown',
            options=list(range(1980, 2022)),
            value=2020
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='daily-weather-custom-year'), width={'size': 6, 'offset': 0}, lg=6),
        dbc.Col(dcc.Graph(id='corn-yield-month-custom-year'), width={'size': 6, 'offset': 0}, lg=6)
    ]),
])


@app.callback(
    Output('corn-yield-graph', 'figure'),
    Input('year', 'value'))
def update_map_graph(year):
    filtered_corn_yield_by_year = corn_yield_with_fips[corn_yield_with_fips['Year'] == year]
    min_yield = filtered_corn_yield_by_year['Yield'].min()
    max_yield = filtered_corn_yield_by_year['Yield'].max()
    fig = px.choropleth(filtered_corn_yield_by_year, geojson=counties, locations='CountyFIPS', color='Yield',
                        color_continuous_scale="Viridis",
                        range_color=(min_yield, max_yield),
                        scope="usa",
                        hover_data=["StateName", "County", "CountyFIPS"]
                        )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
    Output('corn-yield-year', 'figure'),
    Input('corn-yield-graph', 'clickData')
)
def update_yield_year_graph(clickData):
    if clickData is not None:
        points = clickData['points']

        customData = points[0]['customdata']

        state_ = customData[0]
        county_ = customData[1]
        fips_ = customData[2]

        df = corn_yield_with_fips[corn_yield_with_fips['CountyFIPS'] == fips_]
        fig = px.line(df, x='Year', y='Yield',
                      title=f"History Corn Yield for {county_} in {state_}")

        return fig
    else:
        # default figure for history corn yield plot
        default_corn_yield_df = corn_yield_with_fips[corn_yield_with_fips['CountyFIPS'] == '17019']
        default_corn_yield_fig = px.line(default_corn_yield_df, x='Year', y='Yield',
                                         title='History Corn Yield for Champaign in Illinois')

        return default_corn_yield_fig


@app.callback(
    [Output('county-dropdown', 'options'), Output('county-dropdown', 'value')],
    Input('state-dropdown', 'value')
)
def update_county_dropdown_options(selected_state):
    counties_ = df_county[df_county['StateName'] == selected_state]['County']
    return [{'label': i, 'value': i} for i in counties_], counties_.iloc[0]


def find_fips(state_, county_):
    f1 = df_county['StateName'] == state_
    df_ = df_county[f1]
    f2 = df_['County'] == county_
    df_ = df_[f2]
    fips_ = df_['CountyFIPS'].values[0]
    return fips_


def find_maximum_actual_month(df_):
    df_ = df_.copy()
    df_.fillna(0, inplace=True)
    df_["month"] = df_["date"].apply(lambda x: x.month).copy()
    return df_["month"].values[-1]


def processing_custom_year(df_weather_custom_year_, fips_, growing_year_):
    dates = []
    predicted_yields = []

    for growing_month_ in range(1, 13):
        actual_month_ = reverse_growing_month(growing_month_)

        if actual_month_ >= 11:
            start_year = growing_year_ - 1
        else:
            start_year = growing_year_

        temp_d = datetime.datetime(start_year, actual_month_, 15)

        temp_df = preprocess_by_month(df_weather_custom_year_, growing_month_)

        try:
            linear_model = load(f'../model/cache/linear_models/{fips_}_linear.joblib')
            linear_error_model = load(f'../model/cache/linear_error_models/{fips_}_{actual_month_}.joblib')
            linear_prediction = linear_model.predict(np.array([growing_year_]).reshape(-1, 1))[0]
            liner_error_prediction = linear_error_model.predict(temp_df)[0]
            prediction = linear_prediction + liner_error_prediction
        except:
            prediction = -1

        dates.append(temp_d)

        predicted_yields.append(prediction)

    corn_yield_custom_year = corn_yield_with_fips[corn_yield_with_fips['CountyFIPS'] == fips_]
    corn_yield_custom_year = corn_yield_custom_year[corn_yield_custom_year['Year'] == growing_year_]['Yield']

    if len(corn_yield_custom_year) >= 1:
        corn_yield_custom_year = corn_yield_custom_year.values[0]
    else:
        corn_yield_custom_year = -1

    return dates, predicted_yields, growing_year_, corn_yield_custom_year


# def processing_last_year(df_weather_last_year_, fips_):
#     now = datetime.datetime.now()
#     cur_year = now.year - 2
#
#     dates = []
#     predicted_yields = []
#
#     for gm in range(1, 12 + 1):
#         am = reverse_growing_month(gm)
#
#         if am >= 11:
#             start_year = cur_year - 1
#         else:
#             start_year = cur_year
#
#         d = datetime.datetime(start_year, am, 15)
#
#         temp_df = preprocess_by_month(df_weather_last_year_, gm)
#
#         linear_model = load(f'../model/cache/linear_models/{fips_}_linear.joblib')
#         linear_error_model = load(f'../model/cache/linear_error_models/{fips_}_{am}.joblib')
#
#         linear_prediction = linear_model.predict(np.array([cur_year]).reshape(-1, 1))[0]
#         liner_error_prediction = linear_error_model.predict(temp_df)[0]
#         prediction = linear_prediction + liner_error_prediction
#
#         dates.append(d)
#         predicted_yields.append(prediction)
#
#     corn_yield_last_year = corn_yield_with_fips[corn_yield_with_fips['CountyFIPS'] == fips_]
#     corn_yield_last_year = corn_yield_last_year[corn_yield_last_year['Year'] == cur_year]['Yield'].values[0]
#
#     return dates, predicted_yields, cur_year, corn_yield_last_year


def processing(df_weather_until_last_month_, fips_):
    last_actual_month = find_maximum_actual_month(df_weather_until_last_month_)
    last_growing_month = actual_to_growing_month(last_actual_month)

    now = datetime.datetime.now()
    cur_year = now.year

    dates = []
    predicted_yields = []

    for gm in range(1, last_growing_month + 1):
        am = reverse_growing_month(gm)

        if am >= 11:
            start_year = now.year - 1
        else:
            start_year = now.year

        d = datetime.datetime(start_year, am, 15)

        temp_df = preprocess_by_month(df_weather_until_last_month_, gm)

        linear_model = load(f'../model/cache/linear_models/{fips_}_linear.joblib')
        linear_error_model = load(f'../model/cache/linear_error_models/{fips_}_{am}.joblib')

        linear_prediction = linear_model.predict(np.array([cur_year]).reshape(-1, 1))[0]
        liner_error_prediction = linear_error_model.predict(temp_df)[0]
        prediction = linear_prediction + liner_error_prediction

        dates.append(d)
        predicted_yields.append(prediction)

    return dates, predicted_yields


@app.callback(
    [Output('daily-weather-custom-year', 'figure'), Output('corn-yield-month-custom-year', 'figure')],
    [Input('state-dropdown', 'value'), Input('county-dropdown', 'value'), Input('history-year-dropdown', 'value')]
)
def update_county_custom_weather(state_, county_, custom_year_):
    fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
    fig_yields = go.Figure()

    fips_ = find_fips(state_, county_)
    df_weather_custom_year = read_county_weather_custom_year(fips_, custom_year_)

    fig_weather.add_trace(
        go.Scatter(x=df_weather_custom_year['date'], y=df_weather_custom_year['avgt'], mode="lines",
                   name="Average Temperature", opacity=0.5,
                   line=dict(color='firebrick', width=4)), secondary_y=False
    )
    fig_weather.add_trace(
        go.Scatter(x=df_weather_custom_year['date'], y=df_weather_custom_year['pcpn'], mode="lines",
                   name="Precipitation", opacity=0.8,
                   line=dict(color='royalblue', width=4)), secondary_y=True
    )
    fig_weather.update_layout(xaxis_title="Time", yaxis_title="Temperature (F)",
                              title=f"Daily Weather and Yield Prediction in {custom_year_}")
    fig_weather.update_yaxes(title_text="Precipitation (inches)", secondary_y=True)

    dates, predicted_yields, _, corn_yield_custom_year = processing_custom_year(df_weather_custom_year, fips_,
                                                                                custom_year_)

    fig_yields = go.Figure(data=go.Scatter(x=dates, y=predicted_yields, mode='markers', name='predicted yields',
                                           marker=dict(
                                               color='yellow',
                                               size=20,
                                               line=dict(
                                                   color='green',
                                                   width=4
                                               )
                                           ),
                                           ))

    fig_yields.update_xaxes(title_text='Time')
    fig_yields.update_yaxes(title_text="Corn Yield Prediction (BU/ACRE)", range=[50, 250])

    if corn_yield_custom_year >= 0:
        fig_yields.add_trace(
            go.Scatter(x=[datetime.datetime(custom_year_, 11, 1)], y=[corn_yield_custom_year], mode="markers",
                       name="Actual Yield", opacity=0.8,
                       marker=dict(
                           color='blue',
                           size=20,
                           line=dict(
                               color='green',
                               width=4
                           )
                       ), )
        )

    # try:
    #     fips_ = find_fips(state_, county_)
    #     df_weather_custom_year = read_county_weather_custom_year(fips_, custom_year_)
    #
    #     fig_weather.add_trace(
    #         go.Scatter(x=df_weather_custom_year['date'], y=df_weather_custom_year['avgt'], mode="lines", name="Average Temperature", opacity=0.5,
    #                    line=dict(color='firebrick', width=4)), secondary_y=False
    #     )
    #     fig_weather.add_trace(
    #         go.Scatter(x=df_weather_custom_year['date'], y=df_weather_custom_year['pcpn'], mode="lines", name="Precipitation", opacity=0.8,
    #                    line=dict(color='royalblue', width=4)), secondary_y=True
    #     )
    #     fig_weather.update_layout(xaxis_title="Time", yaxis_title="Temperature (F)", title=f"Daily Weather and Yield Prediction in {custom_year_}")
    #     fig_weather.update_yaxes(title_text="Precipitation (inches)", secondary_y=True)
    #
    #     dates, predicted_yields, _, corn_yield_custom_year = processing_custom_year(df_weather_custom_year, fips_, custom_year_)
    #
    #     fig_yields = go.Figure(data=go.Scatter(x=dates, y=predicted_yields, mode='markers', name='predicted yields',
    #                              marker=dict(
    #                                  color='yellow',
    #                                  size=20,
    #                                  line=dict(
    #                                      color='green',
    #                                      width=4
    #                                  )
    #                              ),
    #                              ))
    #
    #     fig_yields.update_xaxes(title_text='Time')
    #     fig_yields.update_yaxes(title_text="Corn Yield Prediction (BU/ACRE)", range=[50, 250])
    #
    #     if corn_yield_custom_year >= 0:
    #         fig_yields.add_trace(
    #             go.Scatter(x=[datetime.datetime(custom_year_, 11, 1)], y=[corn_yield_custom_year], mode="markers", name="Actual Yield", opacity=0.8,
    #                        marker=dict(
    #                            color='blue',
    #                            size=20,
    #                            line=dict(
    #                                color='green',
    #                                width=4
    #                            )
    #                        ),)
    #         )
    # except:
    #     fig_weather.update_layout(title=f"Sorry no model can be found for county: {county_} in {state_}")

    return fig_weather, fig_yields


# @app.callback(
#     [Output('daily-weather-lastyear', 'figure'), Output('corn-yield-month-lastyear', 'figure')],
#     [Input('state-dropdown', 'value'), Input('county-dropdown', 'value')]
# )
# def update_county_lastyear_weather(state_, county_):
#     fips_ = find_fips(state_, county_)
#     df_weather_last_year = read_county_weather_last_year(fips_)
#
#     fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
#     fig_weather.add_trace(
#         go.Scatter(x=df_weather_last_year['date'], y=df_weather_last_year['avgt'], mode="lines", name="Average Temperature", opacity=0.5,
#                    line=dict(color='firebrick', width=4)), secondary_y=False
#     )
#     fig_weather.add_trace(
#         go.Scatter(x=df_weather_last_year['date'], y=df_weather_last_year['pcpn'], mode="lines", name="Precipitation", opacity=0.8,
#                    line=dict(color='royalblue', width=4)), secondary_y=True
#     )
#     fig_weather.update_layout(xaxis_title="Time", yaxis_title="Temperature (F)", title="Daily Weather and Yield Prediction in 2020")
#     fig_weather.update_yaxes(title_text="Precipitation (inches)", secondary_y=True)
#
#     dates, predicted_yields, cur_year, corn_yield_last_year = processing_last_year(df_weather_last_year, fips_)
#
#     fig_yields = go.Figure(data=go.Scatter(x=dates, y=predicted_yields, mode='markers', name='predicted yields',
#                              marker=dict(
#                                  color='yellow',
#                                  size=20,
#                                  line=dict(
#                                      color='green',
#                                      width=4
#                                  )
#                              ),
#                              ))
#
#     fig_yields.update_xaxes(title_text='Time')
#     fig_yields.update_yaxes(title_text="Corn Yield Prediction (BU/ACRE)", range=[50, 250])
#
#     fig_yields.add_trace(
#         go.Scatter(x=[datetime.datetime(cur_year, 11, 1)], y=[corn_yield_last_year], mode="markers", name="Actual Yield", opacity=0.8,
#                    marker=dict(
#                        color='blue',
#                        size=20,
#                        line=dict(
#                            color='green',
#                            width=4
#                        )
#                    ),)
#     )
#
#     return fig_weather, fig_yields


@app.callback(
    [Output('daily-weather-now', 'figure'), Output('corn-yield-month', 'figure')],
    [Input('state-dropdown', 'value'), Input('county-dropdown', 'value')]
)
def update_county_realtime_weather(state_, county_):
    fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
    fig_yields = go.Figure()

    fips_ = find_fips(state_, county_)
    df_weather_until_last_month = read_county_weather_until_last_month(fips_)

    fig_weather.add_trace(
        go.Scatter(x=df_weather_until_last_month['date'], y=df_weather_until_last_month['avgt'], mode="lines",
                   name="Average Temperature", opacity=0.5,
                   line=dict(color='firebrick', width=4)), secondary_y=False
    )
    fig_weather.add_trace(
        go.Scatter(x=df_weather_until_last_month['date'], y=df_weather_until_last_month['pcpn'], mode="lines",
                   name="Precipitation", opacity=0.8,
                   line=dict(color='royalblue', width=4)), secondary_y=True
    )
    fig_weather.update_layout(xaxis_title="Time", yaxis_title="Temperature (F)",
                              title="Daily Weather and Yield Prediction from Last Harvest")
    fig_weather.update_yaxes(title_text="Precipitation (inches)", secondary_y=True)

    dates, predicted_yields = processing(df_weather_until_last_month, fips_)
    fig_yields = go.Figure(data=go.Scatter(x=dates, y=predicted_yields, mode='markers', name='predicted yields',
                                           marker=dict(
                                               color='yellow',
                                               size=20,
                                               line=dict(
                                                   color='green',
                                                   width=4
                                               )
                                           ),
                                           ))

    fig_yields.update_xaxes(title_text='Time')
    fig_yields.update_yaxes(title_text="Corn Yield Prediction (BU/ACRE)", range=[50, 250])

    # try:
    #     fips_ = find_fips(state_, county_)
    #     df_weather_until_last_month = read_county_weather_until_last_month(fips_)
    #
    #     fig_weather.add_trace(
    #         go.Scatter(x=df_weather_until_last_month['date'], y=df_weather_until_last_month['avgt'], mode="lines", name="Average Temperature", opacity=0.5,
    #                    line=dict(color='firebrick', width=4)), secondary_y=False
    #     )
    #     fig_weather.add_trace(
    #         go.Scatter(x=df_weather_until_last_month['date'], y=df_weather_until_last_month['pcpn'], mode="lines", name="Precipitation", opacity=0.8,
    #                    line=dict(color='royalblue', width=4)), secondary_y=True
    #     )
    #     fig_weather.update_layout(xaxis_title="Time", yaxis_title="Temperature (F)", title="Daily Weather and Yield Prediction from Last Harvest")
    #     fig_weather.update_yaxes(title_text="Precipitation (inches)", secondary_y=True)
    #
    #     dates, predicted_yields = processing(df_weather_until_last_month, fips_)
    #     fig_yields = go.Figure(data=go.Scatter(x=dates, y=predicted_yields, mode='markers', name='predicted yields',
    #                              marker=dict(
    #                                  color='yellow',
    #                                  size=20,
    #                                  line=dict(
    #                                      color='green',
    #                                      width=4
    #                                  )
    #                              ),
    #                              ))
    #
    #     fig_yields.update_xaxes(title_text='Time')
    #     fig_yields.update_yaxes(title_text="Corn Yield Prediction (BU/ACRE)", range=[50, 250])
    # except:
    #     fig_weather.update_layout(title=f"Sorry no model can be found for county: {county_} in {state_}")

    return fig_weather, fig_yields


if __name__ == '__main__':
    app.run_server(debug=True)
