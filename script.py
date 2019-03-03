import numpy as np 
import pandas as pd

gs = pd.read_csv('gs.csv', sep=',', header=0, parse_dates=[0], dayfirst=True)
gs.set_index(['Date'])
p = pd.read_csv('p.csv', sep=',', header=0, parse_dates=[0])


p['wg'] = 0
p['ws'] = 0
p['y1wg'] = 0
p['y2wg'] = 0
p['y1ws'] = 0
p['y2ws'] = 0

for index, row in p.iterrows():
    date = row['date']
    lowDate, upperDate = date.split('-')
    in_range_df = gs[gs["Date"].between(lowDate, upperDate)]
    # in_range_df = gs[gs["Date"].isin(pd.date_range(lowDate, upperDate))]
    if len(in_range_df) > 0:
        
        g_mean = in_range_df['g'].mean()
        s_mean = in_range_df['s'].mean()
        p.loc[index, 'wg'] = g_mean
        p.loc[index, 'ws'] = s_mean
        p.loc[index, 'y1wg'] = row['y1'] / g_mean
        p.loc[index, 'y2wg'] = row['y2'] / g_mean
        p.loc[index, 'y1ws'] = row['y1'] / s_mean
        p.loc[index, 'y2ws'] = row['y1'] / s_mean
