import os
import re
from scipy.spatial import distance
from sklearn.preprocessing import Normalizer
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from TFM_Granatiero_Utils.TFM_Pre_Process import *

############################################
#Helper functions to make Content Alg works#
############################################

def get_id_to_check(restaurant_df, data):
    '''Taking just restaurants from Madrid and in common with data'''
    local_restaurant_df = restaurant_df[restaurant_df.city == 4.]
    set_madrid_init     = set(local_restaurant_df.name)
    rest_to_check       = set(data.name_string_value).intersection(set_madrid_init)
    id_to_check         = set(local_restaurant_df.set_index('name').loc[list(rest_to_check),:].id)
    #manually removing one of the two Kappo
    id_to_check = id_to_check - {482}
    
    return id_to_check

def get_restaurant_df(restaurant_df, data, features_cols):
    '''Creating a restaurant df with given features'''
    local_restaurant_df = restaurant_df.copy()
    #Taking just restaurants from Madrid and in common with data
    id_to_check         = get_id_to_check(restaurant_df, data)
    local_restaurant_df = local_restaurant_df.set_index('id').loc[list(id_to_check),:]#.reset_index()
    
    price_num_columns = []
    pattern = r'\d{2,3}'
    price = [re.findall(pattern, s) for s in local_restaurant_df['price']]
    
    for money in price:
        if len(money)==1:
            price_num_columns.append(float(money[0]))
        if len(money)==2:
            price_num_columns.append((float(money[0])+float(money[1]))/2)
        
    local_restaurant_df['price_num'] = price_num_columns
    #creating price range feature
    for idx in local_restaurant_df.index:
        if local_restaurant_df.loc[idx,'price_num'] <= 20:
            local_restaurant_df.loc[idx,'price_num'] = '< 20'
            continue
        if 20 < local_restaurant_df.loc[idx,'price_num'] <= 40:
            local_restaurant_df.loc[idx,'price_num'] = '20-40'
            continue
        if 40 < local_restaurant_df.loc[idx,'price_num'] <= 60:
            local_restaurant_df.loc[idx,'price_num'] = '40-60'
            continue
        if 60 < local_restaurant_df.loc[idx,'price_num'] <= 80:
            local_restaurant_df.loc[idx,'price_num'] = '60-80'
            continue
        if 80 < local_restaurant_df.loc[idx,'price_num'] <= 100:
            local_restaurant_df.loc[idx,'price_num'] = '80-100'
            continue
        if local_restaurant_df.loc[idx,'price_num'] > 100:
            local_restaurant_df.loc[idx,'price_num'] = '> 100'
            continue
        
    Items_df          = local_restaurant_df[features_cols]
    #creating stars feature
    Items_df['stars']  = Items_df['stars'].fillna(0)
    
    return Items_df 

def one_hot_cat(cat_df, col_name):
    '''one-hot-encode of a given feature for restaurant df'''
    local_cat_df = cat_df.copy()
    local_cat_df['True'] = 1 
    local_cat_df = local_cat_df.pivot(index="restaurant_id", columns=col_name, values='True').fillna(0)
    local_cat_df.columns = [col_name+' '+str(i) for i in local_cat_df.columns]
    #Normalizing on the number of occurencies
    sums = local_cat_df.sum(axis=1)
    local_cat_df = local_cat_df.div(sums,axis=0)
    return local_cat_df

def one_hot_restaurant(Items_data):
    Items_data['stars'] = [str(i) for i in Items_data['stars']]
    Items_data['neighborhood'] = [str(i) for i in Items_data['neighborhood']]
    #Items_df['city'] = [str(float(i)) for i in Items_df['city']]
    return pd.get_dummies(Items_data)

def get_user_df_content(user_id, Items_df, User_Items_df):
    '''get a user vector based on restaurants he liked'''
    user_checked_restaurant = User_Items_df.loc[user_id,:][User_Items_df.loc[user_id,:]!=0]

    users_features = pd.DataFrame(columns=Items_df.columns, index=[user_id])
    users_features.loc[user_id,:] = np.zeros(len(Items_df.columns))
    
    for i in user_checked_restaurant.index:
        to_add = Items_df.loc[i,:][Items_df.loc[i,:]!=0].index
        users_features.loc[user_id,to_add]=users_features.loc[user_id,to_add]+user_checked_restaurant.loc[i]
        
    food_sum = users_features.loc[:,'food-type_id 1':'food-type_id 25'].sum(axis=1).values[0] 
    if food_sum!=0:
        users_features.loc[:,'food-type_id 1':'food-type_id 25'] = \
            users_features.loc[:,'food-type_id 1':'food-type_id 25'].divide(food_sum)
                  
    vibe_sum = users_features.loc[:,'vibe_id 5':'vibe_id 35'].sum(axis=1).values[0]
    if vibe_sum!=0:        
        users_features.loc[:,'vibe_id 5':'vibe_id 35'] = \
                users_features.loc[:,'vibe_id 5':'vibe_id 35'].divide(vibe_sum)
            
    star_sum = users_features.loc[:,'stars_0.0':'stars_3.0'].sum(axis=1).values[0]
    if star_sum!=0:            
        users_features.loc[:,'stars_0.0':'stars_3.0'] = \
            users_features.loc[:,'stars_0.0':'stars_3.0'].divide(star_sum)

    neigh_sum = users_features.loc[:,'neighborhood_24.0':'neighborhood_54.0'].sum(axis=1).values[0]
    if neigh_sum!=0:                      
        users_features.loc[:,'neighborhood_24.0':'neighborhood_54.0'] = \
            users_features.loc[:,'neighborhood_24.0':'neighborhood_54.0'].divide(neigh_sum)
        
    price_sum = users_features.loc[:,'price_num_20-40':'price_num_> 100'].sum(axis=1).values[0]
    if price_sum!=0:         
        users_features.loc[:,'price_num_20-40':'price_num_> 100'] = \
            users_features.loc[:,'price_num_20-40':'price_num_> 100'].divide(price_sum)
    
    return users_features

def get_user_df_content_filters(user_id, data, Items_df, User_Items_df,\
                                users_dictionary_inv, vibes_dict_en, vibes_dict, neighbo_dict, types_dict):
    '''get a user vector based on restaurant he liked plus filters he selected'''
    #Initializing
    user_checked_restaurant = User_Items_df.loc[user_id,:][User_Items_df.loc[user_id,:]!=0]
    users_features = pd.DataFrame(columns=Items_df.columns, index=[user_id])
    users_features.loc[user_id,:] = np.zeros(len(Items_df.columns))

    #Adding filters info
    data_HF, data_MFF, data_FA = data[data.event_name=='HOME_FILTERS'], data[data.event_name=='MY_FAVS_FILTER_ADDED'],\
                                 data[data.event_name=='FILTER_ADDED']
    user_HF  = data_HF[data_HF.user_pseudo_id==users_dictionary_inv[user_id]].vibes_string_value
    user_MFF = data_MFF[data_MFF.user_pseudo_id==users_dictionary_inv[user_id]].type_string_value
    user_FA  = data_FA[data_FA.user_pseudo_id==users_dictionary_inv[user_id]].type_string_value

    user_HF, user_MFF, user_FA = list(user_HF.dropna()), list(user_MFF.dropna()), list(user_FA.dropna())
    #A list of all user's filters
    user_filters = user_HF + user_MFF + user_FA
    #Checking if there are no filters
    if len(user_filters)==0:
        return users_features
    #Cleaning emoji to match
    regex = re.compile('[^\w€\b]')
    cleaned_user_filters = [regex.sub('', i) for i in user_filters]
    #local cleaned dictionary
    cleaned_vibes_dict_en   = {regex.sub('', i):vibes_dict_en[i] for i in vibes_dict_en.keys()}
    cleaned_vibes_dict_es   = {regex.sub('', i):vibes_dict[i] for i in vibes_dict.keys()}
    cleaned_neighbo_dict_es = {regex.sub('', i):neighbo_dict[i] for i in neighbo_dict.keys()}
    cleaned_types_dict_es   = {regex.sub('', i):types_dict[i] for i in types_dict.keys()}
    all_filters             = list(cleaned_vibes_dict_en.keys())+list(cleaned_vibes_dict_es.keys())\
                            +list(cleaned_neighbo_dict_es.keys())+list(cleaned_types_dict_es.keys()) 
    cleaned_user_filters    = [x for x in cleaned_user_filters if x in all_filters]
    N_user_filters = len(cleaned_user_filters)
    #Checking if there are no filters
    if N_user_filters==0:
        return users_features
    #Converting to id
    user_filter_ids = []
    for i in cleaned_user_filters:
        if i in set(cleaned_neighbo_dict_es.keys()):
            user_filter_ids.append('neighborhood_'+str(cleaned_neighbo_dict_es[i])+'.0')
        elif i in set(cleaned_types_dict_es.keys()):
            user_filter_ids.append('food-type_id '+str(cleaned_types_dict_es[i]))    
        elif i in set(cleaned_vibes_dict_en.keys()):
            user_filter_ids.append('vibe_id '+str(cleaned_vibes_dict_en[i]))
        elif i in set(cleaned_vibes_dict_es.keys()):
            user_filter_ids.append('vibe_id '+str(cleaned_vibes_dict_es[i]))
        else:
            print(i)
    #Finally updating the score with the filters
    for i in user_filter_ids:
        users_features.loc[user_id,i] = users_features.loc[user_id,i] + 1/N_user_filters
     
    #Adding the rates contents
    for i in user_checked_restaurant.index:
        to_add = Items_df.loc[i,:][Items_df.loc[i,:]!=0].index
        users_features.loc[user_id,to_add]=users_features.loc[user_id,to_add]+user_checked_restaurant.loc[i]
    
    #Normalizing each category
    food_sum = users_features.loc[:,'food-type_id 1':'food-type_id 25'].sum(axis=1).values[0] 
    if food_sum!=0:
        users_features.loc[:,'food-type_id 1':'food-type_id 25'] = \
            users_features.loc[:,'food-type_id 1':'food-type_id 25'].divide(food_sum)
                  
    vibe_sum = users_features.loc[:,'vibe_id 5':'vibe_id 35'].sum(axis=1).values[0]
    if vibe_sum!=0:        
        users_features.loc[:,'vibe_id 5':'vibe_id 35'] = \
                users_features.loc[:,'vibe_id 5':'vibe_id 35'].divide(vibe_sum)
            
    star_sum = users_features.loc[:,'stars_0.0':'stars_3.0'].sum(axis=1).values[0]
    if star_sum!=0:            
        users_features.loc[:,'stars_0.0':'stars_3.0'] = \
            users_features.loc[:,'stars_0.0':'stars_3.0'].divide(star_sum)

    neigh_sum = users_features.loc[:,'neighborhood_24.0':'neighborhood_54.0'].sum(axis=1).values[0]
    if neigh_sum!=0:                      
        users_features.loc[:,'neighborhood_24.0':'neighborhood_54.0'] = \
            users_features.loc[:,'neighborhood_24.0':'neighborhood_54.0'].divide(neigh_sum)
        
    price_sum = users_features.loc[:,'price_num_20-40':'price_num_> 100'].sum(axis=1).values[0]
    if price_sum!=0:         
        users_features.loc[:,'price_num_20-40':'price_num_> 100'] = \
            users_features.loc[:,'price_num_20-40':'price_num_> 100'].divide(price_sum)    
    
    return users_features 

def get_dict_official_id_restaurant(reastaurant_df, data):
    '''official to unofficial restaurant_id'''
    filtered_to_madrid = reastaurant_df.set_index('id').loc[list(get_id_to_check(reastaurant_df, data)),:]
    return {i:filtered_to_madrid.loc[i,'name'] for i in filtered_to_madrid.index}

def get_dict_unofficial_to_official(reastaurant_df, rest_dict, data):
    '''unofficial to official restaurant_id'''
    off_id_dic = get_dict_official_id_restaurant(reastaurant_df, data)
    return {rest_dict[off_id_dic[i]]:i for i in off_id_dic.keys()}

def get_User_Item_Matrix(df_score, min_rated_restaurants=1, min_rated_users=1):    
    '''User Item matrix given a score df with rated'''
    all_users       = set(df_score.loc[:,'user_id'].unique())
    all_restaurants = set(df_score.index.unique())
    
    df_User_Items   = pd.DataFrame(index=all_users, columns=all_restaurants)
    
    for user in all_users:
        scores                    = df_score[df_score.loc[:,'user_id'] == user].score
        df_User_Items.loc[user,:] = scores
        
    col_to_check    = df_User_Items.columns[df_User_Items.count() >= min_rated_restaurants]
    sdf_User_Items = df_User_Items.loc[:,col_to_check]
        
    row_to_check    = df_User_Items.count(axis=1) >= min_rated_users
    df_User_Items = df_User_Items.loc[row_to_check,:]        
        
    return df_User_Items

def get_User_Items_bin(df_score_bin, min_rated_restaurants=1, min_rated_users=1):        
    '''User Item matrix given a binary score df'''
        
    UI           = df_score_bin.pivot(values="is_positive", index="user_id", columns="restaurant_id")*1
    
    col_to_check = UI.columns[UI.count() >= min_rated_restaurants]
    UI           = UI.loc[:,col_to_check]
            
    row_to_check = UI.count(axis=1) >= min_rated_users
    UI           = UI.loc[row_to_check,:]
    
    return UI.fillna(0)

#####################
#Content Based class#
#####################

class Content_Based_Recc:
    """ Collaborative filtering using a custom sim(u,u'). """
    
    def __init__(self, DataFrame, User_Items, restaurant_df, rest_types_df, \
                 rest_vibes_df, types_df, vibes_df, neighbo_df):
        
        """ Constructor """
        self.restaurant_df              = restaurant_df
        self.data                       = DataFrame
        self.User_Items                 = User_Items.copy()
        self.unoff_User_Items_cols      = User_Items.columns
        self.features_cols              = ['stars','neighborhood','price_num'] #features of Item vec
        self.rest_types_df              = rest_types_df  
        self.rest_vibes_df              = rest_vibes_df
        self.all_restaurants            = DataFrame['name_string_value'].unique() 
        self.all_users                  = DataFrame['user_pseudo_id'].unique()    
        self.madrid_restaurants         = set(self.restaurant_df[self.restaurant_df.city == 4.].name)
        self.restuarant_ids             = range(len(self.all_restaurants))
        self.user_ids                   = range(len(self.all_users))
        self.restaurants_dictionary     = dict(zip(self.all_restaurants, self.restuarant_ids))
        self.restaurants_dictionary_inv = {v: k for k, v in self.restaurants_dictionary.items()}
        self.users_dictionary           = dict(zip(self.all_users, self.user_ids))
        self.users_dictionary_inv       = {v: k for k, v in self.users_dictionary.items()}
        self.neighbo_dict               = {neighbo_df.name_es[i]:neighbo_df.id[i] for i in neighbo_df.index} 
        self.vibes_dict                 = {vibes_df.name_es[i]:vibes_df.id[i] for i in vibes_df.index} 
        self.vibes_dict_en              = {vibes_df.name_en[i]:vibes_df.id[i] for i in vibes_df.index}   
        self.types_dict                 = {types_df.name_es[i]:types_df.id[i] for i in types_df.index} 
        self.types_dict_en              = {types_df.name_en[i]:types_df.id[i] for i in types_df.index} 
        self.dict_id_unoff2off          = get_dict_unofficial_to_official(self.restaurant_df, self.restaurants_dictionary, self.data)
        self.dict_id_off2unoff          = {v: k for k, v in self.dict_id_unoff2off.items()}
        self.Items_Content_df           = []
        self.User_Content_df            = []
    
    def fit_Items_Content_df(self): 
        """ fit the items df with all restaurants vectors """
        if self.User_Items.columns.equals(self.unoff_User_Items_cols): 
            self.User_Items.columns = [self.dict_id_unoff2off[i] for i in self.User_Items.columns]
            
        restaurant_data         = get_restaurant_df(self.restaurant_df, self.data, self.features_cols)
        one_hot_restaurant_df   = one_hot_restaurant(restaurant_data)
        one_hot_types           = one_hot_cat(self.rest_types_df, "food-type_id")
        one_hot_vibes           = one_hot_cat(self.rest_vibes_df, "vibe_id")
        Items_df                = pd.concat([one_hot_types,one_hot_vibes,one_hot_restaurant_df], axis=1).dropna()
        
        self.Items_Content_df   = Items_df

    def fit_User_Content_df(self):
        """ fit the users df with all users vectors """
        
        if self.User_Items.columns.equals(self.unoff_User_Items_cols): 
            self.User_Items.columns = [self.dict_id_unoff2off[i] for i in self.User_Items.columns]

        user_df = pd.DataFrame(index=set(self.User_Items.index), columns=self.Items_Content_df.columns)

        for user in set(self.User_Items.index):
            user_df.loc[user,:] = get_user_df_content(user, self.Items_Content_df, self.User_Items).loc[user,:]
            
        self.User_Content_df = user_df 
    
    def fit_User_Content_df_filters(self):
        """ fit the user df with all user vectors considering filters"""
        
        if self.User_Items.columns.equals(self.unoff_User_Items_cols): 
            self.User_Items.columns = [self.dict_id_unoff2off[i] for i in self.User_Items.columns]

        user_df = pd.DataFrame(index=set(self.User_Items.index), columns=self.Items_Content_df.columns)

        for user in set(self.User_Items.index):
            user_df.loc[user,:] = get_user_df_content_filters(user, data, self.Items_Content_df,\
                                                              self.User_Items, self.users_dictionary_inv,\
                                                              self.vibes_dict_en, self.vibes_dict,\
                                                              self.neighbo_dict, self.types_dict).loc[user,:]
            
        self.User_Content_df    = user_df 
    
    def normalize_content_df(self):
        self.Items_Content_df.iloc[:,:]      = Normalizer(norm='l2').fit_transform(self.Items_Content_df)
        self.User_Content_df.iloc[:,:]       = Normalizer(norm='l2').fit_transform(self.User_Content_df)
    
    def get_rate_counts(self):
        rate_counts = {}
        for col in self.User_Items.columns:
            rate_col = dict(self.User_Items[col].value_counts())
            rate_counts = {k: rate_col.get(k, 0) + rate_counts.get(k, 0) for k in set(rate_col) | set(rate_counts)}
        
        plt.bar(rate_counts.keys(), rate_counts.values(), color='g')
        print('The {}% of all possible rates is rated '.format(round((self.User_Items.count().sum())/(self.User_Items.shape[1]*self.User_Items.shape[0]),3)*100))

        return rate_counts
    
    def predict_K_ranked(self, user_id, columns_to_drop=[], K = 5, binary=True):
        """ gives K recommended restaurant for a given user dropping liked ones"""
        
        if binary==False:
            items_to_check = self.User_Items.loc[user_id,:][self.User_Items.loc[user_id,:]<4].index
        
        if binary==True:
            items_to_check = self.User_Items.loc[user_id,:][self.User_Items.loc[user_id,:]==0].index
        
        User_Content_usr    = self.User_Content_df.drop(columns=columns_to_drop).loc[[user_id],:]
        Items_Content_check = self.Items_Content_df.drop(columns=columns_to_drop).loc[items_to_check,:]

        User_Content_usr.iloc[:,:]      = Normalizer(norm='l2').fit_transform(User_Content_usr)
        Items_Content_check.iloc[:,:]   = Normalizer(norm='l2').fit_transform(Items_Content_check)
        
        user_distances = pd.DataFrame(columns=['Distance'], index=items_to_check)
        for rest in items_to_check:
            user_distances.loc[rest,'Distance'] = distance.cosine(User_Content_usr, Items_Content_check.loc[rest,:])
            
        return user_distances.sort_values(by='Distance')[:K]
    
    def full_K_ranked(self, user_id, K = 5):
        """ gives K recommended restaurant for a given user without dropping """
        
        items_to_check = self.User_Items.columns
        user_distances = pd.DataFrame(columns=['Distance'], index=items_to_check)
        for rest in items_to_check:
            user_distances.loc[rest,'Distance'] = distance.cosine(self.User_Content_df.loc[user_id,:], self.Items_Content_df.loc[rest,:])
        return user_distances.sort_values(by='Distance')[:K]

####################################################
#Function with differents metric to evaluate models#
####################################################
    
def prec_recall_k_content(user_id, test_df, predict_method, dict_id_off2unoff, K, col2drop, binary, verbose=False):
    """ Compute precision, recall and accuracy for binary score"""
    
    if user_id not in set(test_df.user_id):
        print('User {} not in test'.format(user_id))
        return np.nan
    
    user_df = test_df[test_df['user_id']==user_id] 
    real_liked = set(user_df.restaurant_id)

    if len(real_liked) == 0:
        return 'Zero likes'
    
    K_acc = len(real_liked)
    top_K_acc = predict_method(user_id, K=K_acc, columns_to_drop=col2drop, binary=binary)
    top_K_acc = [dict_id_off2unoff[i] for i in top_K_acc.index]

    top_K = predict_method(user_id, K=K,columns_to_drop=col2drop, binary=binary)
    top_K = [dict_id_off2unoff[i] for i in top_K.index]    

    number_common_acc = len(set(top_K_acc).intersection(real_liked))
    number_common     = len(set(top_K).intersection(real_liked))
    
    accuracy       = number_common_acc/K_acc
    precision_at_K = number_common/K
    recall_at_K    = number_common/len(real_liked)

    if precision_at_K ==0 and recall_at_K == 0: f1 = 0

    else: f1 = (2*precision_at_K*recall_at_K)/(precision_at_K+recall_at_K)

    if verbose==True:
        toprint=\
        '.....\nThe predicted restaurants are: {}\n.....\nThe real liked are: {}\n.....'.format(top_K\
                                                                                      ,list(real_liked))
        print(toprint)  
        
    return accuracy, f1, precision_at_K, recall_at_K

def mean_prec_recall_at_k_content(User_Items_train, test_df, predict_method,\
                                  dict_id_off2unoff, k, col2drop, binary=True):
    """ Compute precision, recall and accuracy for binary score averaged all users"""
    
    users_to_check = set(User_Items_train.index).intersection(set(test_df.user_id))
    
    scores = [prec_recall_k_content(u, test_df, predict_method, dict_id_off2unoff, k, col2drop, binary=binary) \
                                                               for u in users_to_check]

    scores = [i for i in scores if pd.isnull(i) == False]
    mean_scores = [sum(y) / len(y) for y in zip(*scores)]
    mean_acc = mean_scores[0] ; mean_f1 = mean_scores[1] ; mean_precision = mean_scores[2] ; mean_recall = mean_scores[3]
    User_Tested_Ratio = len(users_to_check)/len(User_Items_train)
    
    return mean_acc, mean_f1, mean_precision, mean_recall, User_Tested_Ratio   

def get_K_restaurant_name(list_of_restaurants_off_id, dict_id_off2unoff, restaurants_dictionary_inv):
    """ get the name of K redstaurants from unofficial ids"""

    rest_names = [dict_id_off2unoff[rist] for rist in list_of_restaurants_off_id.index]
    rest_names = [inv_restaurants_dictionary[rist] for rist in rest_names]
    return rest_names

def scores_bin_content(score_data_bin, restaurants_Datos_Init, full_data,\
                       restaurants__food_types_Datos_Init, restaurants__vibes_Datos_Init,\
                       foodtypes_Datos_Init, vibes_Datos_Init, neighborhoods_Datos_Init, repeat, K, \
                       col2drop,  min_rated_restaurants=1, min_rated_users=1, ts_size=0.1):
    """ Compute precision, recall and accuracy for binary score averaged over all user for several splits"""
   
    tot_scores=[]
    for i in range(repeat): 
        train, test = train_test_split(score_data_bin, test_size=ts_size)
        UI_tr = get_User_Items_bin(train, min_rated_restaurants, min_rated_users)
    
        rec = Content_Based_Recc(full_data, UI_tr, restaurants_Datos_Init, \
                      restaurants__food_types_Datos_Init, restaurants__vibes_Datos_Init,\
                      foodtypes_Datos_Init, vibes_Datos_Init, neighborhoods_Datos_Init)
        
        rec.fit_Items_Content_df()
        rec.fit_User_Content_df()
        tot_scores.append(mean_prec_recall_at_k_content(UI_tr, test, rec.predict_K_ranked, \
                                                        rec.dict_id_off2unoff, K, col2drop, binary=True))
    mean_scores = [sum(y) / len(y) for y in zip(*tot_scores)]
    sigma_acc = np.sqrt(sum([(i[0]-mean_scores[0])**2 for i in tot_scores])/5)
    
    return mean_scores, sigma_acc

def score_content_scored(score_bin,  restaurants_Datos_Init, \
                         restaurants__food_types_Datos_Init, restaurants__vibes_Datos_Init,\
                         foodtypes_Datos_Init, vibes_Datos_Init, \
                         neighborhoods_Datos_Init, filtered_data, col_to_drop, repeat, K=5, ts_size=0.1):
    """ Compute precision, recall and accuracy for rated score averaged over all user for several splits"""
    
    tot_scores=[]
    for i in range(repeat):     
    
        test = train_test_split(score_bin.set_index(['user_id','restaurant_id']), test_size=ts_size)[1]
        test_idx = test.index
    
        idx_to_drop = filtered_data[filtered_data.is_positive==True].reset_index().set_index(['user_id',\
                                                    'restaurant_id']).loc[test_idx,:].set_index('index').index
    
        train_filtered = filtered_data.drop(index=idx_to_drop)
        pre = PreProcessing_Data(data_full, restaurants_df, Rules)
        pre.simple_data = train_filtered
    
        pre.fit_data_score()
        pre.fit_User_Items_Matrix(5,5)
    
        rec = Content_Based_Recc(data_full,  pre.User_Items.fillna(0), restaurants_Datos_Init, \
                      restaurants__food_types_Datos_Init, restaurants__vibes_Datos_Init,\
                      foodtypes_Datos_Init, vibes_Datos_Init, neighborhoods_Datos_Init)
    
        rec.fit_Items_Content_df()
        rec.fit_User_Content_df()

        tot_scores. append(mean_prec_recall_at_k_content(pre.User_Items.fillna(0), test.reset_index(),\
                           rec.predict_K_ranked, rec.dict_id_off2unoff, K, col_to_drop, binary=False))
        
            
    mean_scores = [sum(y) / len(y) for y in zip(*tot_scores)]
    sigma_acc = np.sqrt(sum([(i[0]-mean_scores[0])**2 for i in tot_scores])/5)
    
    return mean_scores, sigma_acc

def liked_statistics(test_df):
    
    def liked_counter(user_id, test_df):
        user_df = test_df[test_df['user_id']==user_id] 
        real_liked = set(user_df.restaurant_id)
        return len(real_liked)
    
    return pd.Series([liked_counter(i, test_df) for i in set(test_df.user_id)]).describe()
