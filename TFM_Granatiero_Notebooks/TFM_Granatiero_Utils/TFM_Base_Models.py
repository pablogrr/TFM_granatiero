############################
############################
## RANDOM RECCOMENDER
############################
############################

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

def get_User_Items_bin(df_score_bin, min_rated_restaurants=1, min_rated_users=1, Normalize=False):        
    ''' Get the binary UI from a binary score df '''    
    UI           = df_score_bin.pivot(values="is_positive", index="user_id", columns="restaurant_id")*1
    
    col_to_check = UI.columns[UI.count() >= min_rated_restaurants]
    UI           = UI.loc[:,col_to_check]
            
    row_to_check = UI.count(axis=1) >= min_rated_users
    UI           = UI.loc[row_to_check,:]
    
    if Normalize==True:
        UI.iloc[:,:] = Normalizer(norm='l2').fit_transform(UI.fillna(0))
        return UI
  
    return UI.fillna(0)


def random_prec_recall_k(user_id, User_Items_train ,test_df, K, verbose=False):
    ''' Performances of a Random Recommender for a given user '''    

    if user_id not in set(test_df.user_id):
        print('User {} not in test'.format(user_id))
        return np.nan
    
    user_df = test_df[test_df['user_id']==user_id] 
    real_liked = set(user_df.restaurant_id)
    all_rest = list(User_Items_train.loc[user_id,:][User_Items_train.loc[user_id,:]==0].index)
    
    top_K = random.sample(all_rest,K)
    K_acc = len(real_liked)
    top_K_acc = random.sample(all_rest,K_acc)

    number_common     = len(set(top_K).intersection(real_liked))
    number_common_acc = len(set(top_K_acc).intersection(real_liked))
    
    accuracy = number_common_acc/K_acc
    precision_at_K = number_common/K
    recall_at_K = number_common/len(real_liked)
    f1 = (2*precision_at_K*recall_at_K)/(precision_at_K+recall_at_K)

    if verbose==True:
        toprint=\
        '.....\nThe predicted restaurants are: {}\n.....\nThe real liked are: {}\n.....'.format(top_K\
                                                                                      ,list(real_liked))
        print(toprint)  
        
    return accuracy, f1, precision_at_K, recall_at_K

def mean_prec_recall_at_k_random(User_Items_train, test_df, all_rest, K):
    ''' Performances of a Random Recommender averaging over all users '''    
    
    users_to_check = set(User_Items_train.index).intersection(set(test_df.user_id))
    
    scores = [random_prec_recall_k(u, User_Items_train, test_df, K) for u in users_to_check]

    scores = [i for i in scores if pd.isnull(i) == False]
    mean_scores = [sum(y) / len(y) for y in zip(*scores)]
    mean_acc = mean_scores[0] ; mean_f1 = mean_scores[1] ; mean_precision = mean_scores[2] ; mean_recall = mean_scores[3]
    User_Tested_Ratio = len(users_to_check)/len(User_Items_train)
    
    return mean_acc, mean_f1, mean_precision, mean_recall, User_Tested_Ratio   


def random_scores_bin(score_data_bin, repeat, K, min_rated_restaurants=1, min_rated_users=1, ts_size=0.1):
    ''' Performances of a Random Recommender averaging over N splits '''    
    
    tot_scores=[]
    for i in range(repeat): 
        train, test = train_test_split(score_data_bin, test_size=ts_size)
        UI_tr = get_User_Items_bin(train, min_rated_restaurants, min_rated_users)

        tot_scores.append(mean_prec_recall_at_k_random(UI_tr, test, list(UI_tr.columns), K))
    mean_scores = [sum(y) / len(y) for y in zip(*tot_scores)]
    sigma_acc = np.sqrt(sum([(i[0]-mean_scores[0])**2 for i in tot_scores])/5)
    
    return mean_scores, sigma_acc 


def liked_statistics(test_df):
    
    def liked_counter(user_id, test_df):
        user_df = test_df[test_df['user_id']==user_id] 
        real_liked = set(user_df.restaurant_id)
        return len(real_liked)
    
    return pd.Series([liked_counter(i, test_df) for i in set(test_df.user_id)]).describe()



#######################
## POPULAR RECCOMENDER#
#######################



def pop_prec_recall_k(user_id, User_Items_train ,test_df, K, verbose=False):
    ''' Performances of a Pop Recommender for a given user '''    
    
    if user_id not in set(test_df.user_id):
        print('User {} not in test'.format(user_id))
        return np.nan
    
    user_df = test_df[test_df['user_id']==user_id] 
    real_liked = set(user_df.restaurant_id)
    all_rest = list(User_Items_train.loc[user_id,:][User_Items_train.loc[user_id,:]==0].index)
    pop_rest = list(User_Items_train.loc[:,all_rest].sum(axis=0).sort_values(ascending=False).index)
    
    top_K = pop_rest[:K]
    K_acc = len(real_liked)
    top_K_acc = pop_rest[:K_acc]

    number_common     = len(set(top_K).intersection(real_liked))
    number_common_acc = len(set(top_K_acc).intersection(real_liked))
    
    accuracy = number_common_acc/K_acc
    precision_at_K = number_common/K
    recall_at_K = number_common/len(real_liked)

    f1 = (2*precision_at_K*recall_at_K)/(precision_at_K+recall_at_K)

    if verbose==True:
        toprint=\
        '.....\nThe predicted restaurants are: {}\n.....\nThe real liked are: {}\n.....'.format(top_K\
                                                                                      ,list(real_liked))
        print(toprint)  
        
    return accuracy, f1, precision_at_K, recall_at_K

def mean_prec_recall_at_k_pop(User_Items_train, test_df, all_rest, K):
    ''' Performances of a Pop Recommender averaging over all users '''    
    
    users_to_check = set(User_Items_train.index).intersection(set(test_df.user_id))
    
    scores = [pop_prec_recall_k(u, User_Items_train, test_df, K) for u in users_to_check]

    scores = [i for i in scores if pd.isnull(i) == False]
    mean_scores = [sum(y) / len(y) for y in zip(*scores)]
    mean_acc = mean_scores[0] ; mean_f1 = mean_scores[1] ; mean_precision = mean_scores[2] ; mean_recall = mean_scores[3]
    User_Tested_Ratio = len(users_to_check)/len(User_Items_train)
    
    return mean_acc, mean_f1, mean_precision, mean_recall, User_Tested_Ratio   


def pop_scores_bin(score_data_bin, repeat, K, min_rated_restaurants=1, min_rated_users=1, ts_size=0.1):
    ''' Performances of a Pop Recommender averaging over N splits '''    
    
    tot_scores=[]
    for i in range(repeat): 
        train, test = train_test_split(score_data_bin, test_size=ts_size)
        UI_tr = get_User_Items_bin(train, min_rated_restaurants, min_rated_users)

        tot_scores.append(mean_prec_recall_at_k_pop(UI_tr, test, list(UI_tr.columns), K))
    mean_scores = [sum(y) / len(y) for y in zip(*tot_scores)]
    sigma_acc = np.sqrt(sum([(i[0]-mean_scores[0])**2 for i in tot_scores])/5)
    
    return mean_scores, sigma_acc


def top_K_pop(user_id, UI, K):
    ''' Pop Recomm for a given user '''    
    
    all_rest = list(UI.loc[user_id,:][UI.loc[user_id,:]==0].index)
    pop_rest = list(UI.loc[:,all_rest].sum(axis=0).sort_values(ascending=False).index)
    
    top_K = pop_rest[:K]
        
    return top_K

def pop_counter(UI, K):
    ''' It counts the different restaurants '''    
    
    total_rest = []
    for u in UI.index:
        total_rest.append(top_K_pop(u, UI, K))
    
    return total_rest
