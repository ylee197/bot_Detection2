import pandas as pd
import numpy as np
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

from multiprocessing import Pool
import math

class Features:
    def __init__(self, df_total, df_diff):
        self.df_total = df_total
        self.df_diff = df_diff
        self.df, self.l_id = self.features(self.df_total, self.df_diff)
        self.df_coord, self.df_code = self.creating_matrix(self.df, self.l_id)
        
    def features(self, df_total, df_diff):
        df_diff = df_diff.reset_index(drop = False)

        l_id = df_diff['id'].unique()

        df_total = df_total[['tid','parent_tweet_id','screen_name','retweet_to','timestamp','host']]
        df_total.columns = ['tid','unwound_url','screen_name','retweet_to','timestamp','host']
        df_total = df_total.explode('unwound_url')
        df_total = df_total.reset_index(drop = True)

        df_total2 = df_total.copy()
        df_total2.loc[:,'first_time'] = df_total2.groupby('unwound_url')['timestamp'].transform('min')
        df_total2.loc[:,'id_list'] = df_total2.groupby('unwound_url')['screen_name'].transform(lambda x: [x.tolist()] * len(x))

        df_total2 = df_total2[df_total2['screen_name'].isin(l_id)]

        df_total2 = df_total2[['tid','unwound_url','retweet_to','first_time','timestamp','id_list','host']]

        df_total_no_duplicate = df_total2.drop_duplicates(['first_time','unwound_url'])
        df_total_no_duplicate = df_total_no_duplicate.dropna()
        df_total_no_duplicate = df_total_no_duplicate.reset_index(drop = True)

        df_total_no_duplicate['list_len'] = df_total_no_duplicate['id_list'].str.len()
        df_total_no_duplicate = df_total_no_duplicate[df_total_no_duplicate['list_len'] > 1]
        l_id = df_total[df_total['unwound_url'].isin(df_total_no_duplicate['unwound_url'].tolist())]['screen_name'].tolist()

        return df_total_no_duplicate, l_id

    def intervals(self, group_size, num):
        part_duration = num / group_size
        return [(math.floor(i * part_duration), math.floor((i + 1) * part_duration)) for i in range(group_size)]
    
    def work(self, partition_range):
        START = partition_range[0]
        END = partition_range[1]
        df_split = self.df.iloc[START:END, :]
        
        id_count = df_split.shape[0]
        i = 0
        #l_code = []
        df_pair = pd.DataFrame()
        with tqdm(total = df_split.shape[0]) as pbar:
            for index, row in df_split.iterrows():
                l_row = list(set(row['id_list']))
                #l = [item for item in row['id_list'] if item in gb_id['id'].tolist()]
                l = [item for item in l_row if item in self.l_id]
                #l = list(set(row['id_list']) - (set(row['id_list']) - set(self.l_id)))
                l.sort()
                l_pair = [(a, b) for idx, a in enumerate(l) for b in l[idx+1:]]
                df_summary = pd.DataFrame(l_pair)
                df_pair = pd.concat((df_pair, df_summary), axis = 0)
                pbar.update()
        df_pair.columns = ['ID1','ID2']
        gb_pair = df_pair.groupby(['ID1','ID2']).size()
        gb_pair = gb_pair.reset_index(drop = False)
        gb_pair.columns = ['ID1','ID2','count']
        
        if gb_pair.shape[0] != 0:
            return gb_pair
        else:
            return None
    
    def creating_matrix(self, df, l_id):
        df_id = pd.DataFrame(l_id, columns = ['id'])
        df_id = df_id.reset_index(drop = False)
        gb_id = df_id.groupby('id').size()
        gb_id = gb_id.reset_index(drop = False)
        gb_id.columns = ['id','count']
        gb_id = gb_id[gb_id['count'] > 1]
        self.l_id = gb_id['id'].unique()
        '''
        l_code = []
        df_pair = pd.DataFrame()
        with tqdm(total = df.shape[0]) as pbar:
            for index, row in df.iterrows():
                #l = [item for item in row['id_list'] if item in gb_id['id'].tolist()]
                l = list(set(row['id_list']) - (set(row['id_list']) - set(gb_id['id'].tolist())))
                l.sort()
                l_pair = [(a, b) for idx, a in enumerate(l) for b in l[idx+1:]]
                df_summary = pd.DataFrame(l_pair)
                df_pair = pd.concat((df_pair, df_summary), axis = 0)
                pbar.update()
        df_pair.columns = ['ID1','ID2']

        gb_pair = df_pair.groupby(['ID1','ID2']).size()
        gb_pair = gb_pair.reset_index(drop = False)
        gb_pair.columns = ['ID1','ID2','count']
        '''
        num_rows = df.shape[0]
        df_data = pd.DataFrame()
        group_size = 20
        partitions = self.intervals(group_size, num_rows)
        print(partitions)

        partitioned_data = []
        with Pool(processes = group_size) as proc:
            #with tqdm(total=group_size) as pbar:
            for i, page in enumerate(proc.imap_unordered(self.work, partitions)):
                partitioned_data.append(page)
            #        pbar.update()
            df_data = df_data.append(partitioned_data)
            if df_data.shape[0] == 0:
                print('there are not enough shares!')
        gb_pair = df_data.groupby(['ID1','ID2'])['count'].sum()
        gb_pair = gb_pair.reset_index(drop = False)
        gb_pair.columns = ['ID1','ID2','count']
        return(gb_pair, df)

'''
def features(df_SWM, df_diff):
    df_diff = df_diff.reset_index(drop = False)
    print(df_diff)

    l_id = df_diff['id'].unique()
   
    #df_total = df_SWM[['tid','unwound_url','screen_name','retweet_to','timestamp','host']]
    df_total = df_SWM[['tid','parent_tweet_id','screen_name','retweet_to','timestamp','host']]
    df_total.columns = ['tid','unwound_url','screen_name','retweet_to','timestamp','host']
    df_total = df_total.explode('unwound_url')
    df_total = df_total.reset_index(drop = True)

    df_total2 = df_total.copy()
    df_total2.loc[:,'first_time'] = df_total2.groupby('unwound_url')['timestamp'].transform('min')
    df_total2.loc[:,'id_list'] = df_total2.groupby('unwound_url')['screen_name'].transform(lambda x: [x.tolist()] * len(x))
    print(df_total)
    
    df_total2 = df_total2[df_total2['screen_name'].isin(l_id)]
    
    print(df_total2.columns)
    df_total2 = df_total2[['tid','unwound_url','retweet_to','first_time','timestamp','id_list','host']]

    df_total_no_duplicate = df_total2.drop_duplicates(['first_time','unwound_url'])
    df_total_no_duplicate = df_total_no_duplicate.dropna()
    df_total_no_duplicate = df_total_no_duplicate.reset_index(drop = True)

    #df_total_no_duplicate_list_len = df_total_no_duplicate.copy()
    df_total_no_duplicate['list_len'] = df_total_no_duplicate['id_list'].str.len()
    df_total_no_duplicate = df_total_no_duplicate[df_total_no_duplicate['list_len'] > 1]
    l_id = df_total[df_total['unwound_url'].isin(df_total_no_duplicate['unwound_url'].tolist())]['screen_name'].tolist()
    return df_total_no_duplicate, l_id

def SUM(list1, list2):
    final_list = list1 + list2
    return final_list

## output marked IDs which are paired(retweet together)
def id_convert_index(l, l_id):
    df_id = pd.DataFrame(l_id, columns = ['id'])
    df_id = df_id.reset_index(drop = False)
    df_id['index'] = df_id['index'] + 1
    l_new = []
    df_id['mark'] = 0
    for index in range(len(l)):
        df_id.loc[df_id['id'] == l[index], 'mark'] = df_id['index']
    l_new = df_id['mark'].tolist()
    l_new = [i for i in l_new if i != 0]
    return(l_new)

def matrix(l, l_2d):
    #print(l)
    for i in range(len(l)):
        for j in range(len(l)):
            if j > i:
                l_2d[l[i]][l[j]] = l_2d[l[i]][l[j]] + 1
    return l_2d

def creating_matrix(df_URL, l_id):
    df_id = pd.DataFrame(l_id, columns = ['id'])
    df_id = df_id.reset_index(drop = False)
    gb_id = df_id.groupby('id').size()
    gb_id = gb_id.reset_index(drop = False)
    gb_id.columns = ['id','count']
    gb_id = gb_id[gb_id['count'] > 1]
    #dict_id = pd.Series(df_id['index'].values, index = df_id['id']).to_dict()
    #print(dict_id)
    l_code = []
    df_pair = pd.DataFrame()
    id_unique = set(gb_id['id'].tolist())
    with tqdm(total = df_URL.shape[0]) as pbar:
        for index, row in df_URL.iterrows():
            id_list = list(set(row['id_list']))
            l = [item for item in id_list if item in id_unique]
            #l = list(set(row['id_list']) - (set(row['id_list']) - set(gb_id['id'].tolist())))
            l.sort()
            print(l)
            l_pair = [(a, b) for idx, a in enumerate(l) for b in l[idx+1:]]
            print(l_pair)
            df_summary = pd.DataFrame(l_pair)
            #print(df_summary.shape)
            df_pair = pd.concat((df_pair, df_summary), axis = 0)
            pbar.update()
    print(df_pair)
    #df_code = df_URL.copy()
    #print('code columns')
    #print(df_code[['id_list','code_list']])
    df_pair.columns = ['ID1','ID2']
    gb_pair = df_pair.groupby(['ID1','ID2']).size()
    gb_pair = gb_pair.reset_index(drop = False)
    gb_pair.columns = ['ID1','ID2','count']
   
    return(gb_pair, df_URL)
'''
'''    
def creating_matrix_2(df_URL):
    ################
    # df_URL에 있는 unwound_url을 retweet한 모든 screen_name이것을 찾는게 더 빠른지 test해볼것. 
    # id_list를 id_dict로 바꾸어 index로 바꾼후, matrix을 거치지 않고 바로 pair of ids로 바꾸어 보지
    # test_list = [1, 7, 4, 3]
    # res = [(a, b) for idx, a in enumerate(test_list) for b in test_list[idx+1:]}
    #################
    #df_URL = pd.read_pickle('output/tweets_w_minTime_IDlist_1s.pkl')
    print(df_URL)
    print(df_URL.columns)
    
    #df_URL = df_URL.head(1000)
    df_ID = df_URL[['id_list']]
    print(df_ID)
    print(df_ID.shape[0])
    df_original = df_ID.copy()
    while(df_original.shape[0] > 1):
        df_new = pd.DataFrame(columns = ['id_list'])
        print(len(df_original)//2)
        ## For odd length dataframe
        if len(df_original)%2 ==1:
            new_list = df_original.loc[len(df_original)//2, 'id_list']
            df_temp = pd.DataFrame(columns = ['id_list'])
            df_temp.loc[0, 'id_list'] = new_list
            df_new = df_new.append(df_temp)

        for i in range(len(df_original)//2): 
            length = df_original.shape[0]-1
            first = df_original.loc[i, 'id_list']
            second = df_original.loc[length-i, 'id_list']
            print(first)
            print(second)
            new_list = SUM(first, second)
            df_temp = pd.DataFrame(columns = ['id_list'])
            df_temp.loc[0, 'id_list'] = new_list
            df_new = df_new.append(df_temp)
    #        print(df_new)
        df_new = df_new.reset_index(drop = True)
        df_original = df_new.copy()
        print(df_new)
    l_original_id = df_original.loc[0, 'id_list']
    id_len = len(l_original_id)
    print(id_len)

    # I who is seen more than 2 times
    df_list = pd.DataFrame(l_original_id, columns = ['ID'])
    gb_list = df_list.groupby('ID').size()
    gb_list = gb_list.reset_index(drop = False)
    gb_list.columns = ['ID','count']
    gb_list = gb_list[gb_list['count']>1]

    l_id = list(set(gb_list['ID'].tolist()))
    print(len(l_id))

    l_2d = [[0]*(len(l_id)+1) for i in range((len(l_id)+1))]
    total_url = df_URL.shape[0]
    with tqdm(total = df_URL.shape[0]) as pbar:
        for i in range(len(df_URL)):
            pbar.update(1)
            print(f"processing {i} of {total_url}, url = {df_URL.loc[i,'unwound_url']}")
            id_list = df_URL.loc[i,'id_list']
            index_list = id_convert_index(id_list, l_id)
            print(index_list)
            l_2d = matrix(index_list, l_2d)

    df_2d = pd.DataFrame(l_2d)
    df_2d.to_csv('output/original_matrix_no_duplicate_unwound_1s_slow.csv', index = False)

    l_id = np.insert(l_id,0,0)
    df_2d['index'] = l_id

    df_id = pd.DataFrame(l_id).T

    new_df_2d = df_id.append(df_2d)
    new_df_2d = new_df_2d.set_index('index')
    new_df_2d = new_df_2d.reset_index(drop = False)
    new_df_2d = new_df_2d.drop(columns = [0])
    new_df_2d = new_df_2d.drop([1,1])
    print(new_df_2d)
    new_df_2d.to_csv('output/matrix_no_duplicate_unwound_1s_slow.csv', index = False)
    
    return new_df_2d

def matrix2DF(df_matrix):
    #df_matrix = pd.read_csv('/home/ylee197/YJ/bot_detection/2022/0322/output/matrix_no_duplicate_unwound_1s_slow.csv',dtype='unicode', index_col = 0)
    df_matrix = df_matrix.reset_index(drop = False)
    print(df_matrix)
    sys.exit(0)
    l_coord =[]
    i = 0
    for row_index, row in df_matrix.iterrows():
        if row_index == 0:
            l_id = row
        else:
            print(row_index, row)
            for col_index, item in enumerate(row):
                if col_index >= row_index:
                    #print(row_index)
                    #print(col_index)
                    ID1 = row[0]
                    ID2 = l_id[col_index]
                    count = item
                    print("%s, %s, %s" %(row[0],l_id[col_index], item))
                    list_item = {'ID1' : ID1, 'ID2' : ID2, 'count' : count}
                    if item != '0':
                        l_coord.append(list_item)
                    #if col_index == 10:
                    #    print(l_coord)
                    #    df_coord = pd.DataFrame(l_coord)
                    #    print(df_coord)
    #    sys.exit(0)
    df_coord = pd.DataFrame(l_coord)
    df_coord['count'] = df_coord['count'].astype(int)
    print(df_coord)
    df_coord.to_csv('output/coordination_unwound_w0_1s.csv', index = False)
    
    return df_coord
'''

