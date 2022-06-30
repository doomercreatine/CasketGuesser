"""_summary_
This file is for an interactive dashboard to visualize the guesses that people have made to date.
"""
from dash import Dash, html, dcc, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
import math
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Master Casket Predictions"
server = app.server

tables = pd.read_csv("tables.csv")
    
with open("item_prices.json", "r") as f:
    data2 = json.loads(f.read())
    
with open("item_ids.json", "r") as f:
    items = json.loads(f.read())

def get_rates(df):
    rolls = []
    for _, row in df.iterrows():
        for i in range(row['Weight']):
            rolls.append(row['Item'])
    return rolls

def roll_tables(table):
    rolling = True
    str_roll = ""
    tbl = table
    while rolling:
        new_tables = get_rates(tables[tables['Table'] == tbl])
        str_roll = np.random.choice(new_tables)
        if tables.query(f'Table == "{tbl}" and Item == "{str_roll}"')['Amount'].values[0] != 'Table':
            rolling = False
        else:
            tbl = str_roll
    return str_roll

starting = get_rates(tables[tables['Table'] == 'Start'])
mimic = get_rates(tables[tables['Table'] == 'Mimic'])

# Set up the app layout
app.layout = html.Div([
    html.H1("Master Casket Reward Predictor", style={'textAlign': 'center'}), 
    # Leaderboard header and datatable formatting
    dcc.Store(id="store"),  # store that holds the data reference
    dcc.Store(id="sha", data=""),
    dcc.Interval(
            id='interval',
            interval=300*1000,
            n_intervals=0),
    dash_table.DataTable(
        id="prediction",
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="single",
        column_selectable=False,
        row_selectable=False,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        cell_selectable=False,
        page_action="native",
        page_current= 0,
        page_size= 10,
        filter_options={'case': 'insensitive'},
        style_cell={'textAlign': 'center', 'background': '#222'}
    ),
    html.Button('New Prediction', id='submit-val', n_clicks=0, style={'marginLeft': '50%'})
])


@app.callback(
    [Output("prediction", "columns"),
    Output("prediction", "data")], 
    Input("store", "data"),
    Input('submit-val', 'n_clicks'),)
def show_data(data, n_clicks):
    loot_results = []
    num_rolls = np.random.choice([5,6,7]) # Determines how many loot rolls will be made
    print(f"Number of rolls: {num_rolls}")
    each_rolls = [] # Holds the initial table that each loot roll hit
    for i in range(0, num_rolls): # Iterate over each roll to pick a starting table
        each_rolls.append(np.random.choice(starting))
    pet_roll = np.random.choice(get_rates(tables[tables['Table'] == 'Pet Table']))
    mimic_roll = np.random.choice(mimic)
    
    
    full_loots = {}
    r = 1
    for roll in each_rolls:
        new_tables = get_rates(tables[tables['Table'] == roll])
        str_roll = np.random.choice(new_tables)
        if tables.query(f'Table == "{roll}" and Item == "{str_roll}"')['Amount'].values[0] == 'Table':
            str_roll = roll_tables(str_roll)
        amt = tables[tables['Item'] == str_roll]['Amount'].values[0]
        bonus = tables[tables['Item'] == str_roll]['Random'].values[0]
        tbl = tables[tables['Item'] == str_roll]['Table'].values[0]
        if "-" in amt:
            split_amt = amt.split("-")
            amt = int(np.random.choice(range(int(split_amt[0]), int(split_amt[1])+1)))
        else:
            amt = int(amt)
        if not math.isnan(bonus):
            amt = amt + np.random.choice(range(0, int(bonus) + 1))
           # if str_roll in full_loots.keys():
            full_loots[r]= {'item': str_roll, 'amount': amt}
        else:
            full_loots[r] = {'item': str_roll, 'amount': amt}
        profit = 0
        if str_roll == 'Coins':
            sell = 1
        else:
            id = items[str_roll]
            sell = data2['data'][f"{id}"]['low']
        profit += sell * full_loots[r]['amount']
        loot_results.append([r, tbl, str_roll, amt, sell, profit])
        r += 1
    loot_df = pd.DataFrame(loot_results, columns=['Roll', 'Table', 'Item', 'Amount', 'Price per each', 'Total'])
    tot_profits = {'Roll':'', 'Table':'', 'Item':'Total Value', 'Amount':'', 'Price per each':'', 'Total':sum(loot_df['Total'].values)}
    loot_df = loot_df.append(tot_profits, ignore_index=True)
    loot_df['Amount'] = ['{:,}'.format(int(item)) if item != '' else '' for item in loot_df['Amount']]
    loot_df['Price per each'] = ['{:,}'.format(int(item)) if item != '' else '' for item in loot_df['Price per each']]
    loot_df['Total'] = ['{:,}'.format(item) for item in loot_df['Total']]
    loot_df = loot_df.append({'Roll':'', 'Table':'Bloodhound', 'Item':'', 'Amount':'', 'Price per each':'', 'Total': pet_roll}, ignore_index=True)
    loot_df = loot_df.append({'Roll':'', 'Table':'Mimic', 'Item':'', 'Amount':'', 'Price per each':'', 'Total': mimic_roll}, ignore_index=True)
    return(
        [
            [{"name": i, "id": i, "deletable": False, "selectable": True} for i in loot_df.columns],
            loot_df.to_dict('records')
        ]
    )
    




# Run local server
if __name__ == '__main__':
    app.run_server(debug=True)
