import numpy as np
import pandas as pd
import tqdm as tqdm
import torch 
import os

import argparse
import logging
import time

from data_loading import get_dataset
from URL import URL
from fast_retweeted import time_threshold
#from retweet_features import features, creating_matrix, matrix2DF
from retweet_features import Features
from bot_detector import bot, multi

parser = argparse.ArgumentParser('GCN_Bot')
parser.add_argument('--data_path',
                    type = str,
                    help = 'Data path',
                    default = '.')
#parser.add_argument('--log', type = str, help = 'Log level', default='INFO', choices = ['DEBUG', 'INFO', 'WARNING'])
parser.add_argument('--gup_idx', type = int, help = 'Indexes of gpu to run program on', default=0)

def run(args):
    device=torch.device(f'cuda:{args.gup_idx}' if torch.cuda.is_available() else 'cpu')
    '''
    ## Preprocessing tweet dataset
    cwd = os.getcwd()
    path = cwd + '/data/'
    if os.path.isdir(path) == False:
        os.mkdir(path)
    dataset = get_dataset(data_path=args.data_path)
    dataset = dataset.rename(columns = {'tweet_id':'tid', 'created_at':'timestamp'})
    dataset['host'] = ''
    #dataset['screen_name'] = dataset['user_id']
    dataset.to_pickle(f'{args.data_path}/data/preprocessed_whole_retweet.pkl')
    
    dataset = dataset.dropna(subset = ['retweet_to'])
    dataset = dataset[dataset['tweet_type'] == 'retweet']
    print(dataset.columns)
    dataset.to_pickle(f'{args.data_path}/data/preprocessed_retweet.pkl')
    #data = URL(dataset)
    #print(data.df)
   
    #data.df.to_pickle(f'{args.data_path}/data/preprocessed_retweet.pkl')
    data = pd.read_pickle(f'{args.data_path}/data/preprocessed_retweet.pkl')
    data = data.rename(columns = {'tweet_id':'tid', 'user_id':'screen_name','created_at':'timestamp'})
    data['host'] = ''
    df_time = time_threshold(data, args.data_path)
    
    df_time.to_pickle(f'{args.data_path}/data/time_threshold.pkl')
    df_time = pd.read_pickle(f'{args.data_path}/data/time_threshold.pkl')
    df_time = df_time.rename(columns = {'time':'timestamp'})
    print(df_time.columns)
    
    data = pd.read_pickle(f'{args.data_path}/data/preprocessed_retweet.pkl')
    data = data.rename(columns = {'user_id':'screen_name'})
    
    feature = Features(data, df_time)
    feature.df_code.to_pickle(f'{args.data_path}/data/code_list.pkl')
    feature.df_coord.to_pickle(f'{args.data_path}/data/coordinated.pkl')
    
    #df_features, l_id = features(data, df_time) 
    #df_features.to_pickle(f'{args.data_path}/data/feature.pkl')
    #df_coord, df_code = creating_matrix(df_features, l_id)
    #df_code.to_pickle(f'{args.data_path}/data/code_list.pkl')
    #print(df_coord)
    #df_coord.to_pickle(f'{args.data_path}/data/coordinated.pkl')
    '''
    data = pd.read_pickle(f'{args.data_path}/data/preprocessed_retweet.pkl')
    data.to_csv(f'{args.data_path}/data/preprocessed_retweet.csv')
    data = data.dropna(subset = ['retweet_to'])
    data = data.rename(columns = {'user_id':'screen_name'})
    df_time = pd.read_pickle(f'{args.data_path}/data/time_threshold.pkl')
    df_code = pd.read_pickle(f'{args.data_path}/data/code_list.pkl')
    df_coord = pd.read_pickle(f'{args.data_path}/data/coordinated.pkl')
    data['host'] = ''
    coord_bot = bot(df_coord, df_time, data, df_code, args.data_path)
    self_bot = multi(df_code)
    print(self_bot)
    df_bot = pd.concat([coord_bot,self_bot])
    df_bot = df_bot.drop_duplicates()
    print(df_bot)
    df_bot.to_csv(f'{args.data_path}/data/total_bot_list.csv', index = False)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    #logger = logging.getLogger(__name__)
    #logger.setLevel(level=getattr(logging, args.log.upper(), None))
    #logging.basicConfig(filename ='log_file.log', filemode = 'w', level=logging.INFO, force = True)
    run(args)