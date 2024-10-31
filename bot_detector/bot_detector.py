import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

def multi(df):
    l = []
    print(df.columns)
    for index, row in df.iterrows():
        retweet_tid = row['unwound_url']
        list_len = row['list_len']
        num_unique = len(list(set(row['id_list'])))
        unique_id = list(set(row['id_list']))
        #df.loc[index,'num_unique'] = len(list(set(row['id_list'])))
        item = {'retweet_tid':retweet_tid, 'list_len':list_len, 'num_unique':num_unique, 'unique_id':unique_id}
        l.append(item)
    df = pd.DataFrame(l)
    df['num_unique'] = df['num_unique'].astype(int)
    df_unique = df[df['num_unique'] == 1]
    
    for index, row in df_unique.iterrows():
        df_unique.loc[index,'unique'] = row['unique_id'][0]
    
    df_unique = df_unique.sort_values(by = ['list_len'])
    df_bot = df_unique[df_unique['list_len'] > 2]
    if df_bot.shape[0] > 1:
        l_bot = list(set(df_bot['unique_id'].sum()))
        df_multi_bot = pd.DataFrame(l_bot, columns = ['id'])
        df_multi_bot['fast/slow'] = 'multi'
    else:
        df_multi_bot = pd.DataFrame()
    return df_multi_bot

def histogram(df, n_bins, data_path, log, title, TS):
    # Generate a normal distribution, center at x=0 and y=5
    plt.figure(figsize=(15,8))
    x = df['count']

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
    for i in range(0, TS):
        patches[i].set_facecolor('#220690')
    for i in range(TS, n_bins):
        patches[i].set_facecolor('#F0F920')
    
    plt.suptitle('Histogram for retweet count threshold ', fontsize=30)
    plt.grid(axis = 'x')
    plt.xlabel('number of tweets which are retweeted by pair of IDs', fontsize=18)
    plt.ylabel('number of ID pairs', fontsize=18)
    plt.savefig(f'{data_path}/data/histogram_'+title+'.png')
'''    
def ID_domain(df, l_TS):
    l_url = []
    df_pair = pd.DataFrame()
    with tqdm(total = df.shape[0]) as pbar:
        for index, row in df.iterrows():
            l = row['id_list']
            fast_l = list(set(l) - (set(l) - set(l_TS)))
            item = {'tid':row['tid'], 'unwound_url':row['unwound_url'], 'id_list':fast_l, 'host':row['host']} 
            l_url.append(item)
            if len(fast_l) > 1:
                l_pair = [(a, b) for idx, a in enumerate(fast_l) for b in fast_l[idx+1:]]
                df_summary = pd.DataFrame(l_pair)
                df_summary.loc[:, 'url'] = row['unwound_url']
                df_summary.loc[:, 'domain'] = row['host']
                df_pair = pd.concat((df_pair, df_summary), axis = 0)
            pbar.update()
        
    df_url = pd.DataFrame(l_url)
    df_pair = df_pair.reset_index(drop = True)
    df_pair.columns = ['ID1','ID2','url','domain']
    return df_url, df_pair 

def test_domain(df_total, l_slow, l):
    df_domain = pd.DataFrame(l_slow, columns = ['id'])
    
    for index, row in df_domain.iterrows():
        ID = row['id']
        df_summary = df_total[df_total['id'] == ID]
        has_domain = df_summary[df_summary['host'].isin(l)]
        if has_domain.shape[0] > 0:
            df_domain.loc[index, 'test'] = True
        else:
            df_domain.loc[index, 'test'] = False
    return df_domain
'''
def ID_target(df, l_TS):
    l_url = []
    df_pair = pd.DataFrame()
    with tqdm(total = df.shape[0]) as pbar:
        for index, row in df.iterrows():
            l = row['id_list']
            fast_l = list(set(l) - (set(l) - set(l_TS)))
            item = {'tid':row['tid'], 'target':row['unwound_url'], 'id_list':fast_l} 
            #item = {'tid':row['tid'], 'target':row['target'], 'id_list':fast_l, 'host':row['host']} 
            l_url.append(item)
            if len(fast_l) > 2:
                l_pair = [(a, b) for idx, a in enumerate(fast_l) for b in fast_l[idx+1:]]
                df_summary = pd.DataFrame(l_pair)
                df_summary.loc[:, 'target'] = row['unwound_url']
                #df_summary.loc[:, 'domain'] = row['host']
                df_pair = pd.concat((df_pair, df_summary), axis = 0)
            pbar.update()
        
    df_url = pd.DataFrame(l_url)
    df_pair = df_pair.reset_index(drop = True)
    df_pair.columns = ['ID1','ID2','target']
    return df_url, df_pair 

def test_target(df_total, l_slow, l):
    df_target = pd.DataFrame(l_slow, columns = ['user'])
    with tqdm(total = df_target.shape[0]) as pbar:
        for index, row in df_target.iterrows():
            ID = row['user']
            df_summary = df_total[df_total['id'] == ID]
            has_target = df_summary[df_summary['expanded_url'].isin(l)]
            if has_target.shape[0] > 0:
                df_target.loc[index, 'test'] = True
            else:
                df_target.loc[index, 'test'] = False
            pbar.update()
    return df_target

def bot(df, df_diff, df_SWM, df_code, data_path):
    f = open(f'{data_path}/data/report.txt','w')
    df_diff = df_diff.reset_index(drop = False)
    
    df_shares = df_SWM[['screen_name','timestamp','parent_tweet_id','host']]
    #df_shares.columns =['screen_name','timestamp','unwound_url','host']
    df_shares.columns = ['id','time','expanded_url','host']
    df_shares = df_shares.explode('expanded_url')
    df_shares = df_shares.reset_index(drop = True)
    
    ID = pd.concat((df['ID1'], df['ID2']), axis = 0)
    ID = ID.unique()
    l_fast = df_diff['id'].unique().tolist()
    l_slow = list(set(ID) - set(l_fast))
    
    ## fast-fast connection ##
    df_fast = df[(df['ID1'].isin(l_fast) & df['ID2'].isin(l_fast))]
    df_not_fast = df[~(df['ID1'].isin(l_fast) & df['ID2'].isin(l_fast))]
 
    #df_fast['count'] = df_fast['count'].astype(np.int64)
    df_fast = df_fast.reset_index(drop = True)
    df_not_fast = df_not_fast.reset_index(drop = True)
   
    print('total IDs : ' + str(len(ID)))
    print('total fast IDs : ' + str(len(l_fast)))
    print('total slow IDs : ' + str(len(l_slow)))
    print('fast data tuple : ',df_fast.shape)
    print('not fast data tuple : ',df_not_fast.shape)
    print('total data tuple : ',df.shape)
    print('fast max : ' + str(df_fast['count'].max()))
    f.write('\ntotal IDs : {}\n'.format(str(len(ID))))
    f.write('\ntotal fast IDs : {}\n'.format(str(len(l_fast))))
    f.write('\ntotal slow IDs : {}\n'.format(str(len(l_slow))))
    f.write('\ndf_fast shape : {}\n'.format(str(df_fast.shape)))
    f.write('\ndf_fast top5 : {}\n'.format(str(df_fast.head(5))))
    f.write('\ndf_fast count max : {}\n'.format(str(df_fast['count'].max())))
    
    ### Creating a histogram
    #histogram(dataframe, n_bins, file_path, log, title)
    TS = 10
    histogram(df_fast, (df_fast['count'].max())-1, data_path, 'not_log','fast',TS)
    histogram(df_fast[df_fast['count'] <= 50], 49, data_path, 'not_log', 'fast_small',TS)
    histogram(df_fast[df_fast['count'] <= 50], 49, data_path, 'log', 'fast_small_log',TS)
    # # Fast ID 중 4번 이상 같이 retweet한 ID들
    FAST_TS = 13
    
    histogram(df_fast, (df_fast['count'].max())-1, data_path, 'not_log','fast', FAST_TS)
    histogram(df_fast[df_fast['count'] <= 50], 49, data_path, 'not_log', 'fast_small', FAST_TS)
    histogram(df_fast[df_fast['count'] <= 50], 49, data_path, 'log', 'fast_small_log', FAST_TS)
    
    df_fast_TS = df_fast[df_fast['count'] > FAST_TS]
    df_fast_TS = df_fast_TS.reset_index(drop = True)
    print(df_fast_TS)
    f.write('\ndf_fast_more_threshold : {}\n'.format(str(df_fast_TS.shape)))
    
    ## number of IDs
    l_fast_TS = list(set(df_fast_TS['ID1'].tolist() + df_fast_TS['ID2'].tolist()))
    print('number of fast IDs which retweeted more than Threshold : ' + str(len(l_fast_TS)))

    ## slow dataset
    print('number of slow IDs : ' + str(len(l_slow)))
    f.write('\nnumber of slow IDs : {}\n'.format(str(len(l_slow))))

    ## Find the host which retweeted both fast IDs
    df_url, df_pair = ID_target(df_code, l_fast_TS)
    df_pair.to_csv(f'{data_path}/data/domain_fastIDpairs.csv', index = False)
    #print(len(df_pair['domain'].unique()))
    #f.write('\nnumber host retweeted by fast IDs : {}\n'.format(str(len(df_pair['domain'].unique()))))
    
    ## Domain test
    l_target = df_pair['target'].unique().tolist()
    df_tested = test_target(df_shares, l_slow ,l_target)
    
    ## Domain test end
    l_tested = df_tested['user'].tolist()
    
    df_only_slow = df_not_fast[(df_not_fast['ID1'].isin(l_tested) & df_not_fast['ID2'].isin(l_tested))]
    
    # Filtering out the pair which are ID1 and ID2 are both fast_IDs.
    print("####### slow propagation #########")
    gb = df_only_slow.groupby('count').size()
    gb = gb.reset_index(drop = False)
    gb.columns = ['number','count']
    gb = gb.sort_values(by = 'count', ascending = False)
    print(gb)
    
    print('Slow coordinatied retweet count max : '+str(df_only_slow['count'].max()))
    histogram(df_only_slow, df_only_slow['count'].max()-1, data_path, 'not_log','slow', TS)
    histogram(df_only_slow[df_only_slow['count'] <= 50], 49, data_path, 'not_log', 'slow_small', TS)
    histogram(df_only_slow[df_only_slow['count'] <= 50], 49, data_path, 'log', 'slow_small_log', TS)
    # # slow ID
    SLOW_TS = 25
    histogram(df_only_slow, df_only_slow['count'].max()-1, data_path, 'not_log','slow', SLOW_TS)
    histogram(df_only_slow[df_only_slow['count'] <= 50], 49, data_path, 'not_log', 'slow_small', SLOW_TS)
    histogram(df_only_slow[df_only_slow['count'] <= 50], 49, data_path, 'log', 'slow_small_log', SLOW_TS)
    
    df_slow_TS = df_only_slow[df_only_slow['count'] > SLOW_TS]
    df_slow_TS = df_slow_TS.reset_index(drop = True)
    l_slow_TS = list(set(df_slow_TS['ID1'].tolist() + df_slow_TS['ID2'].tolist()))
    print(df_slow_TS.shape)
    
    print('number of slow IDs passing Threshold : ' + str(len(list(set(df_slow_TS['ID1'].tolist() + df_slow_TS['ID2'].tolist())))))
    f.write('\nnumber of slow IDs passing Threshold: {}\n'.format(str(len(list(set(df_slow_TS['ID1'].tolist() + df_slow_TS['ID2'].tolist()))))))

    ######check this number#######
    l_only_slow = list(set(df_only_slow['ID1'].tolist() + df_only_slow['ID2'].tolist()))
    print('number of only slow ID : ' + str(len(l_only_slow)))
   
    print('number of fastIDs that pass the Threshold: ' +str(len(list(set(df_fast_TS['ID1'].tolist() + df_fast_TS['ID2'].tolist())))))
    f.write('\nnumber of fastIDs : {}\n'.format(str(len(list(set(df_fast_TS['ID1'].tolist() + df_fast_TS['ID2'].tolist()))))))
    print('number of slowIDs that pass the Threshold : ' +str(len(list(set(df_slow_TS['ID1'].tolist() + df_slow_TS['ID2'].tolist())))))
    f.write('\nnumber of slowIDs that pass the Threshold : {}\n'.format(str(len(list(set(df_slow_TS['ID1'].tolist() + df_slow_TS['ID2'].tolist()))))))
    
    ### Connection between slow and fast ID ###
    df_connection = df_slow_TS[(df_slow_TS['ID1'].isin(l_fast_TS)&df_slow_TS['ID2'].isin(l_only_slow))|(df_slow_TS['ID1'].isin(l_only_slow)&df_slow_TS['ID2'].isin(l_fast_TS))]
    #print(df_connection)
    print('number of only slow IDs :',len(l_only_slow))
    f.write('\nnumber of only slow IDs : {}\n'.format(str(len(l_only_slow))))
    print('number of only fast IDs :',len(l_fast))
    f.write('\nnumber of only fast IDs : {}\n'.format(str(len(l_fast))))
    l_total = list(set(l_slow_TS + l_fast_TS))
    print('number of total ID : ' + str(len(l_total)))
    f.write('\nnumber of total IDs : {}\n'.format(str(len(l_total))))
    
    #df_total = df[df['ID1'].isin(l_total) & df['ID2'].isin(l_total)]
    df_total = pd.concat((df_slow_TS, df_fast_TS), axis = 0)
    df_total = pd.concat((df_total, df_connection), axis = 0)
    df_total = df_total.reset_index(drop = True)
    
    print('number of fast dataset : ' + str(df_fast_TS.shape))
    f.write('\nnumber of fast dataset : {}\n'.format(str(df_fast_TS.shape)))
    print('number of slow dataset : ' + str(df_slow_TS.shape))
    f.write('\nnumber of slow dataset : {}\n'.format(str(df_slow_TS.shape)))
    print('number of total dataset : ' + str(df_total.shape))
    f.write('\nnumber of total dataset : {}\n'.format(str(df_total.shape)))

    print('total dataset')
    print(df_total.shape)
    print('total IDs\' number : '+str(len(list(set(df_total['ID1'].tolist() + df_total['ID2'].tolist())))))
    l_total = list(set(df_total['ID1'].tolist() + df_total['ID2'].tolist()))
    print('number of total IDs : ' + str(len(l_total)))
    f.write('\ndf_total.shape : {}\n'.format(str(df_total.shape)))
    f.write('\ntotal IDs\' number : {}\n'.format(str(len(l_total))))

    ## For the suspecious ID list
    df_IDlist1 = df_total[['ID1']]
    df_IDlist1.columns = ['id']
    df_IDlist2 = df_total[['ID2']]
    df_IDlist2.columns = ['id']
    df_IDlist = pd.concat((df_IDlist1,df_IDlist2), axis = 0)
    df_IDlist = df_IDlist.reset_index(drop = True)
    gb_IDlist = df_IDlist.groupby('id').size()
    gb_IDlist = gb_IDlist.reset_index(drop = False)
    gb_IDlist.columns = ['id','count']

    gb_IDlist.loc[gb_IDlist['id'].isin(l_fast_TS), 'fast/slow'] = 'Fast'
    gb_IDlist.loc[gb_IDlist['id'].isin(l_only_slow), 'fast/slow'] = 'Slow'
    gb_IDlist.to_csv(f'{data_path}/data/suspicious_ID_list.csv', index = False)
    print(gb_IDlist)
    return(gb_IDlist)
    
