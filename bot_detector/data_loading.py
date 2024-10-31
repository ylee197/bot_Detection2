import numpy as np
import pandas as pd
from tqdm import tqdm
import torch 

import argparse
import logging
import time

import math

def get_dataset(data_path):
    df = pd.read_csv('/Users/sang-pilhan/Desktop/2023/Hasan/2024_Ukrain/2024_08/profile/input/ukraine_2023-07-01_2024-09_30.csv', dtype = str, index_col = 0)
    df.columns = ['tweet_id','parent_tweet_id','tweet_text','tweet_type','created_at','language','user_id','username','parent_user_id','parent_user_name','topic']
    #df_filtered = df[df['langauge'].isin(['en', 'ru', 'uk'])]
    #df2 = pd.read_csv('/home/ylee197/YJ/bot_detection/magi/data/botnet_detection_english_tweets.csv', dtype = str)
    #df3 = pd.read_csv('/home/ylee197/YJ/bot_detection/magi/data/botnet_detection_arabic_tweets.csv', dtype = str)
    ## Merge whole dataset
    #df = df1.copy()
    #df = pd.concat((df, df2), axis = 0)
    #df = pd.concat((df, df3), axis = 0)
    df = df.drop_duplicates()
    df = df.dropna(subset = ['user_id'])
    df = df.reset_index(drop = True)
    print('Whole dataset : '+str(df.shape))
    #df = df.head(2000)
    ## Data organizing
    df['created_at'] = pd.to_datetime(df['created_at'], errors = 'coerce', format = '%Y-%m-%d %H:%M:%S')
    df['created_at'] = df['created_at'].astype('datetime64[ns]')

    df['date'] = df['created_at'].dt.date
    df['year'] = df['created_at'].dt.year
    print(df.groupby('tweet_type').size())
    print(df[df['tweet_type'].isnull()].shape)
    print(df.groupby('year').size())
    
    ## collecting tweet dataset with tweet_to information
    df_retweet = df[df['parent_tweet_id'].isin(df['tweet_id'].to_list())]
    print('dataset with tweet_to : '+str(df_retweet.shape))
    print(df_retweet.groupby('tweet_type').size())
    
    dic_id = pd.Series(df['user_id'].values, index = df['tweet_id']).to_dict()
    df = df[df['parent_tweet_id'].notnull()]
    df = df[df['tweet_text'].notnull()]
    print(df.shape)
    #df1 = df1.head(2000)
    with tqdm(total = df.shape[0]) as pbar:
        for index, row in df.iterrows():
            #print(dic_id['tweet_id'])
            if row['parent_tweet_id'] in dic_id.keys() :
                #print(dic_id[row['parent_tweet_id']])
                df.loc[index,'retweet_to'] = dic_id[row['parent_tweet_id']]
                
            pbar.update()
    #df_retweet = df7[df7['retweet_to'].notnull()]
    df['tweet_text'] = df['tweet_text'].str.replace('\n',' ')
    df['tweet_text'] = df['tweet_text'].str.replace('\r',' ')
    return df
    #df_only_retweet = df_retweet[df_retweet['tweet_type'] == 'retweet']
    #df_only_retweet.to_csv('output/Ru_war_retweet_to_0220.csv', index = False)