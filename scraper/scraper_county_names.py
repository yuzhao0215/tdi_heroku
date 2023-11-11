import requests
import json
import pickle

key_api = '11710A85-AC22-3A4C-9856-61BF8D6BD047'

states_names = ['Illinois',
                'Indiana',
                'Iowa',
                'Kansas',
                'Kentucky',
                'Michigan',
                'Minnesota',
                'Missouri',
                'Nebraska',
                'North Dakota',
                'Ohio',
                'South Dakota',
                'Wisconsin']

states_alphas = ['IL', 'IN', 'IA', 'KS', 'KY', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']

states_alpha2name = dict(zip(states_alphas, states_names))

states_alpha2county = dict()

for state_alpha, state_name in states_alpha2name.items():
    params = dict(key=key_api,
                  state_alpha=state_alpha,
                  param='county_name'
                  )

    res = requests.get('http://quickstats.nass.usda.gov/api/get_param_values',
                       params=params)

    if res.status_code == requests.codes.ok:
        counties = res.json()['county_name']
        states_alpha2county[state_alpha] = counties
    else:
        print("Error request url")

f = open('county_names_dic.pkl', "wb")
pickle.dump(states_alpha2county, f)
f.close()

print(states_alpha2county.items())