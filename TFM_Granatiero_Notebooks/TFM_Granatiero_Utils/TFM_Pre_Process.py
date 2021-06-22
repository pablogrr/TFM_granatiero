import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random


##############################################
#Functions to be used in the preprocess class#
##############################################

def event_mask(df):
    """ it filters the relevant features """
    return df[[(df.loc[:,'event_name']  == 'view_item') 
             | (df.loc[:,'event_name']  == 'RESTAURANT_ACTION') 
             | (df.loc[:,'event_name']  == 'RESTAURANT_FAVOURITE') 
             | (df.loc[:,'event_name']  == 'MY_FAVS_REMOVE_RESTAURANT')
             | (df.loc[:,'event_name']  == 'CARD_SWIPE')
             | (df.loc[:,'event_name']  == 'RESTAURANT_BLACKLISTED')][0]] 

def get_user_df(user_id, df, mask):
    """ it slices the df of a given user """    
    data_user = df[df.loc[:,'user_id'] == user_id].drop(columns=['user_id', 'user_pseudo_id'])
    return mask(data_user)

def Row_Score(row, data_user, Score_Dic):
    """ it gives the score of a row based on a score dictionary rules """     
    if data_user.loc[row,:].event_name == 'CARD_SWIPE':
        row_score = Score_Dic[data_user.loc[row,:].event_name][data_user.loc[row,:].dir_string_value]
        return row_score 

    if data_user.loc[row,:].event_name == 'RESTAURANT_ACTION':
        row_score = Score_Dic[data_user.loc[row,:].event_name][data_user.loc[row,:].action_string_value]
        return row_score 
    
    else:
        row_score = Score_Dic[data_user.loc[row,:].event_name]
        return row_score

def get_user_score_max(user_id, df, Score_Dic, min_num_swipe, swipe):
    """ it gives the max score for each rest of a given user """     
    data_user = df[df.loc[:,'user_id'] == user_id]
    User_Scores = [Row_Score(row, data_user, Score_Dic) for row in data_user.index]
    data_user['Score'] = User_Scores
    user_score_df = data_user.pivot(columns='restaurant_id', values='Score')
    final_score = {}
    #If we consinder swipe we can score only a minimum number of swipes
    if swipe==True:
        for col in user_score_df.columns: 
            if set(data_user[data_user.loc[:,'restaurant_id'] == col].event_name.values) == {'CARD_SWIPE'} and \
                 set(data_user[data_user.loc[:,'restaurant_id'] == col].dir_string_value) == {'LEFT'}:
            
                if len(user_score_df.loc[:,col].dropna()) >= min_num_swipe:
                    final_score[col] = Score_Dic['CARD_SWIPE']['LEFT']
        
            else:
                final_score[col] = user_score_df.loc[:,col].max()
    
        return pd.Series(final_score) 
    
    if swipe==False:
        for col in user_score_df.columns: 
            final_score[col] = user_score_df.loc[:,col].max()
        return pd.Series(final_score) 



def get_User_Item_Matrix(df_score, min_rated_restaurants=1, min_rated_users=1):    
    """ it gives a UI matrix having as input a score dataframe """        
    all_users       = set(df_score.loc[:,'user_id'].unique())
    all_restaurants = set(df_score.index.unique())
    
    df_User_Items   = pd.DataFrame(index=all_users, columns=all_restaurants)
    
    for user in all_users:
        scores                    = df_score[df_score.loc[:,'user_id'] == user].score
        df_User_Items.loc[user,:] = scores
    #We can consider a minimum number of likes for rests and users    
    col_to_check    = df_User_Items.columns[df_User_Items.count() >= min_rated_restaurants]
    sdf_User_Items = df_User_Items.loc[:,col_to_check]
        
    row_to_check    = df_User_Items.count(axis=1) >= min_rated_users
    df_User_Items = df_User_Items.loc[row_to_check,:]        
        
    return df_User_Items

def check_only_swipe(simple_data):
    """ it checks if the only interaction between a rest and a user is swipe """            
    users = set([u if set(simple_data[simple_data.user_id==u].event_name)=={'CARD_SWIPE'} \
             else np.nan for u in set(simple_data.user_id)])
    
    users = [x for x in users if str(x) != 'nan']
    
    onlies = [u if set(simple_data[simple_data.user_id==u].dir_string_value)=={'LEFT'} \
              else np.nan for u in set(simple_data.user_id)]
    
    return [x for x in onlies if str(x) != 'nan']

def user_actions_counter(simple_data, mini, maxi):
    """ it filters a minimum and a maximum number of interactions for each user """            
    usr_action_counts = simple_data.pivot_table(values="event_timestamp", index="user_id", aggfunc=pd.Series.nunique).fillna(0)
    rare = list(usr_action_counts[usr_action_counts < mini].dropna().index)
    outliers = list(usr_action_counts[usr_action_counts > maxi].dropna().index)
    return rare+outliers

def check_zero_positive(simple_data):
    """ it finds all user with zero likes """        
    positive = []

    for i in simple_data.index:  
        if simple_data.event_name[i] == 'RESTAURANT_FAVOURITE':
            positive.append(True)
        elif simple_data.action_string_value[i] == 'call' or  simple_data.action_string_value[i] == 'delivery'\
             or simple_data.action_string_value[i] == 'book_url' or simple_data.action_string_value[i] == 'favourite_press':
            positive.append(True)
        elif simple_data.dir_string_value[i] == 'DOWN':
            positive.append(True)
        else:
            positive.append(False)
     
    positive = pd.Series(positive, index=simple_data.user_id)
    
    zero_positive_idx = []
    
    for i in set(positive.index):
        if type(positive[i]) == pd.Series:
            if sum(positive[i])==0:
                zero_positive_idx.append(i)
        elif type(positive[i]) == np.bool_:
            zero_positive_idx.append(i)
        else: continue
    return zero_positive_idx


#Our score rules
Rules = {'CARD_SWIPE'         :{'LEFT':1, 'DOWN':4}, 
              'RESTAURANT_ACTION'   :{'delivery':5,'book_url':5,'chefInstagram':3,'instagram':3, 
                                      'menu':3,'website':3,'maps':3,'call':5, 'favourite_press':4, 
                                      'curated_by':3},
              'RESTAURANT_FAVOURITE':4,
              'view_item':2}

##########################
#PreProcessing_Data class#
##########################

class PreProcessing_Data:
    """ Collaborative filtering using a custom sim(u,u'). """
    
    def __init__(self, DataFrame, restaurants_df, Rules):
        """ Constructor """
        self.coll_mask          = ['event_name', 'event_timestamp', 'name_string_value', 
                                   'action_string_value', 'dir_string_value', 'user_pseudo_id' ]
        self.restaurants_df     = restaurants_df
        self.df                 = DataFrame.reset_index(drop=True)
        self.simple_data        = []
        self.score_data         = []
        self.score_data_bin     = []
        self.User_Items         = []
        self.User_Items_bin     = []
        self.rules              = Rules 
        self.all_restaurants    = DataFrame['name_string_value'].unique() 
        self.all_users          = DataFrame['user_pseudo_id'].unique()    
        self.madrid_restaurants = set(restaurants_df[restaurants_df.city == 4.].name)
        self.restuarant_ids     = range(len(self.all_restaurants))
        self.user_ids           = range(len(self.all_users))
        self.restaurants_dictionary  = dict(zip(self.all_restaurants, self.restuarant_ids))
        self.users_dictionary        = dict(zip(self.all_users, self.user_ids))
        self.restaurants_dictionary_inv  = {v: k for k, v in self.restaurants_dictionary.items()}
        self.users_dictionary_inv        = {v: k for k, v in self.users_dictionary.items()}        
     
    def init_simple_data(self, events=event_mask):
        '''update simple_data without "not Madrid" restaurants and with unofficials ids. It drops all the filters actions'''
        self.simple_data      = self.df[self.coll_mask]
        self.simple_data      = events(self.simple_data)
        nan_to_drop           = self.simple_data[self.simple_data.name_string_value.isna()].index
        self.simple_data      = self.simple_data.drop(index=nan_to_drop)
        restuarants_to_take   = set(self.simple_data.name_string_value).intersection(self.madrid_restaurants)
        self.simple_data      = self.simple_data.set_index('name_string_value').loc[list(restuarants_to_take),:]
        self.simple_data      = self.simple_data.reset_index()
        
        self.simple_data['restaurant_id'] = [self.restaurants_dictionary[self.simple_data.loc[row,'name_string_value']] 
                                            for row in self.simple_data.index]
    
        self.simple_data['user_id']       = [self.users_dictionary[self.simple_data.loc[row,'user_pseudo_id']] 
                                            for row in self.simple_data.index]        
    
    def filter_data_for_collaborative(self, events=event_mask, swipe=False):
        '''add is_positive column and drops the removed favourites'''            
        #by default we drop all swipe left
        if swipe==False:
            self.simple_data = self.simple_data[self.simple_data.dir_string_value!='LEFT']

        #adding likes column
        for i in self.simple_data.index:
            
            if self.simple_data.loc[i,'event_name']=='RESTAURANT_FAVOURITE' or\
             self.simple_data.loc[i,'dir_string_value']=='DOWN' or\
             self.simple_data.loc[i,'action_string_value']=='favourite_press' or\
             self.simple_data.loc[i,'action_string_value']=='book_url' or\
             self.simple_data.loc[i,'action_string_value']=='delivery' or\
             self.simple_data.loc[i,'action_string_value']=='call':
                self.simple_data.loc[i,'is_positive']=True

            elif self.simple_data.loc[i,'event_name']=='MY_FAVS_REMOVE_RESTAURANT' or\
             self.simple_data.loc[i,'event_name']=='RESTAURANT_BLACKLISTED':    
                self.simple_data.loc[i,'is_positive']=False

        
        #Dropping removed favourites            
        ui_to_drop = self.simple_data[self.simple_data.is_positive==False].set_index(['user_id','restaurant_id']).index
        ui_to_drop = set(ui_to_drop)
        ui_true = self.simple_data[self.simple_data.is_positive==True].set_index(['user_id','restaurant_id']).index
        ui_to_drop_t = ui_to_drop.intersection(set(ui_true))
        ui_to_drop_f = ui_to_drop - ui_to_drop_t
        self.simple_data = self.simple_data.set_index(['user_id','restaurant_id']).drop(index=ui_to_drop_t).reset_index()
        self.simple_data = self.simple_data.set_index(['user_id','restaurant_id']).drop(index=ui_to_drop_f).reset_index()
        
    def clean_anomalies(self, mini, maxi, swipe=False, positive=False):
        '''filters a min and a max number of actions'''
        rare_users       = user_actions_counter(self.simple_data, mini, maxi)
        
        if swipe==True:
            only_swipe       = check_only_swipe(self.simple_data)
            self.simple_data = self.simple_data.set_index('user_id').drop(index=rare_users+only_swipe).reset_index()
        if swipe==False:
            self.simple_data = self.simple_data.set_index('user_id').drop(index=rare_users).reset_index()
        
        if positive==True:
            zero_positive    = check_zero_positive(self.simple_data)
            self.simple_data = self.simple_data.set_index('user_id').drop(index=zero_positive).reset_index()
    
    def fit_data_score_bin(self):
        '''generates a binary score dataframe'''
        indx = self.simple_data.is_positive.dropna().index
        self.score_data_bin = self.simple_data.loc[indx,:][['user_id','restaurant_id','is_positive']].drop_duplicates() 
        
    def fit_User_Items_bin(self, min_rated_restaurants=1, min_rated_users=1, Normalize=False):        
        '''generates a binary User Item Matrix'''
        self.User_Items_bin  = self.score_data_bin.dropna().pivot(values="is_positive", index="user_id", columns="restaurant_id")*1
        col_to_check         = self.User_Items_bin.columns[self.User_Items_bin.count() >= min_rated_restaurants]
        self.User_Items_bin  = self.User_Items_bin.loc[:,col_to_check]
        #We can set a min and max number of likes    
        row_to_check         = self.User_Items_bin.count(axis=1) >= min_rated_users
        self.User_Items_bin  = self.User_Items_bin.loc[row_to_check,:]
        self.User_Items_bin  = self.User_Items_bin.fillna(0)

        if Normalize==True:
                self.User_Items_bin.iloc[:,:] = Normalizer(norm='l2').fit_transform(self.User_Items_bin.fillna(0))

    
    def fit_data_score(self,  min_num_swipe=0, swipe=False):
        '''generates a max score dataframe based on a score rule dictionary'''            
        self.score_data = get_user_score_max(self.simple_data['user_id'].unique()[0], self.simple_data, self.rules, min_num_swipe, swipe).to_frame(name='score')
        self.score_data['user_id'] = self.simple_data['user_id'].unique()[0]
        for user in set(self.simple_data['user_id'].unique()) - {self.simple_data['user_id'].unique()[0]}:
            data_user = get_user_score_max(user, self.simple_data, self.rules, min_num_swipe, swipe).to_frame(name='score')
            data_user['user_id'] = user

            self.score_data      = pd.concat([self.score_data, data_user])
    
    def rescale_score_data(self, min_val=1, max_val=5):
        scaler = MinMaxScaler((min_val, max_val))
        self.score_data['score'] = scaler.fit_transform(self.score_data[['score']])
    
    def fit_User_Items_Matrix(self, min_liked_restaurants=1, min_liked_users=1):
        '''generates a not binary user item matrix'''
        all_users       = set(self.score_data.loc[:,'user_id'].unique())
        all_restaurants = set(self.score_data.index.unique())
    
        self.User_Items = pd.DataFrame(index=all_users, columns=all_restaurants)
    
        for user in all_users:
            scores      = self.score_data[self.score_data.loc[:,'user_id'] == user].score
            self.User_Items.loc[user,:] = scores
        
        to_check = self.User_Items>=4
        
        rest_to_check = to_check.sum(axis=0)[to_check.sum(axis=0)>=min_liked_restaurants].index
        self.User_Items = self.User_Items.loc[:,rest_to_check]

        user_to_check = to_check.sum(axis=1)[to_check.sum(axis=1)>=min_liked_users].index
        self.User_Items = self.User_Items.loc[user_to_check,:]
         
    def get_rate_counts(self):
        '''it gives a plot to monitor the user scores'''
        rate_counts = {}
        for col in self.User_Items.columns:
            rate_col = dict(self.User_Items[col].value_counts())
            rate_counts = {k: rate_col.get(k, 0) + rate_counts.get(k, 0) for k in set(rate_col) | set(rate_counts)}
        
        plt.bar(rate_counts.keys(), rate_counts.values(), color='g')
        print('The {}% of all possible rates is rated '.format(round((self.User_Items.count().sum())/(self.User_Items.shape[1]*self.User_Items.shape[0]),3)*100))

        return rate_counts
    
    def get_rest_count_plot(self):
        '''it gives a plot of restaurants likes'''
        restaurants_count = pd.DataFrame(self.User_Items.count()).sort_values(by=0,ascending=False)

        restaurants_count.plot.bar().xaxis.set_visible(False)
        plt.gcf().set_size_inches(20,10)
        
    def get_user_count_plot(self):
        '''it gives a plot of users' likes'''
        user_count = pd.DataFrame(self.User_Items.count(axis=1)).sort_values(by=0,ascending=False)
        user_count.plot.bar().xaxis.set_visible(False)
        plt.gcf().set_size_inches(20,10)
        
        
    def get_user_action_boxplot(self):
        '''it gives a plot to monitor user actions outliers'''
        usr_action_counts=self.simple_data.pivot_table(values="event_timestamp",\
                                                       index="user_id", aggfunc=pd.Series.nunique).fillna(0)
        
        print(usr_action_counts.describe())
        
        fig = plt.figure(figsize =(10, 7)) 
        plt.boxplot(usr_action_counts.values.reshape(len(usr_action_counts)))
        plt.show()  

