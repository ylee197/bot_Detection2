import pandas as pd
import numpy as np
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import json
from tqdm import tqdm
from multiprocessing import Pool
import math

TIME_THRESHOLD = 5
def histogram(df, n_bins, data_path, log, title):
    # Generate a normal distribution, center at x=0 and y=5
    plt.figure(figsize=(15,8))
    x = df['diff']

    # N is the count in each bin, bins is the lower-limit of the bin
    if log == 'log':
        N, bins, patches = plt.hist(x, bins=n_bins, edgecolor = 'white', linewidth = 1, log = True)
    else:
        N, bins, patches = plt.hist(x, bins=n_bins, edgecolor = 'white', linewidth = 1)
    # We'll color code by height, but you could use any scalar
    
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    '''
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.plasma(norm(thisfrac))
        thispatch.set_facecolor(color)
    '''
    for i in range(0,TIME_THRESHOLD):
        patches[i].set_facecolor('#F0F920')
    for i in range(TIME_THRESHOLD, 20):
        patches[i].set_facecolor('#220690')
    
    plt.suptitle('Histogram for retweet count threshold ', fontsize=30)
    plt.grid(axis = 'x')
    plt.xlabel('seconds', fontsize=18)
    plt.ylabel('number of IDs', fontsize=18)
    plt.savefig(f'{data_path}/data/retweet_interval_ID_'+title+'.png')

def time_threshold(df, data_path):
    #print(df.columns)
    #df_shares = df[['tid','unwound_url','screen_name','retweet_to','timestamp','host']]
    df_shares = df[['tid','parent_tweet_id','screen_name','retweet_to','timestamp','host']]
    df_shares.columns = ['tid','expanded_url','id','retweet_to','time','host']
    #print(df_shares)

    # count the number of different Posts
    df_post = pd.DataFrame(df_shares['expanded_url'].value_counts())
    df_post.reset_index(level=0, inplace=True)
    df_post.columns = ['URL', 'ct_shares']

    # filter the Posts where the count is > 1.
    df_post = df_post[df_post['ct_shares'] > 1]

    # filter the df_shares that join with df_post
    df_shares = df_shares[df_shares.set_index('expanded_url').index.isin(df_post.set_index('URL').index)]

    # metrics creation
    shares_gb = df_shares.groupby(['id'])
    ## Sorting by id and time
    df_sort = df_shares.sort_values(['id','time'], ascending = True)
    df_sort['first_share_date'] = shares_gb['time'].transform('min')

    ## Set the first retweeted time of each retweet post as time from the posted time
    df_sort['diff'] = df_sort['time'].diff()
    df_sort['diff'] = df_sort['diff'].dt.total_seconds()
    df_sort.loc[df_sort['time'] == df_sort['first_share_date'] ,'diff'] = -1
    df_sort.to_pickle('data/time_diff.plk')
    
    df_sort = df_sort[df_sort['diff'] >= 0]

    ## Graph for time differences
    histogram(df_sort, 100, data_path, 'not_log','full')
    histogram(df_sort[df_sort['diff'] < 21], 20, data_path, 'not_log','small')
  
    gb_filtered = df_sort[df_sort['diff'] <= TIME_THRESHOLD]
    print(gb_filtered)
    
    return gb_filtered