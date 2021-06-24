import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import random
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import itertools  

###############################################
#Helper Functions to make Collab properly work#
###############################################

def get_User_Items_bin(df_score_bin, min_rated_restaurants=1, min_rated_users=1, Normalize=False):        
    '''get user item binary matrix'''    
    UI           = df_score_bin.pivot(values="is_positive", index="user_id", columns="restaurant_id")*1
    
    col_to_check = UI.columns[UI.count() >= min_rated_restaurants]
    row_to_check = UI.count(axis=1)[UI.count(axis=1) >= min_rated_users].index
    
    UI           = UI.loc[row_to_check,col_to_check]
    
    if Normalize==True:
        UI.iloc[:,:] = Normalizer(norm='l2').fit_transform(UI.fillna(0))
        return UI

    
    return UI.fillna(0)
        
def get_User_Item_Matrix(df_score, min_rated_restaurants=1, min_rated_users=1):    
    '''get user item matrix with scores'''        
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

def features_interaction(simple_data, string, min_count):
    '''count a give feature interaction with a minimum count'''        
    
    if string!='view_item':

        interact_count = simple_data[simple_data.action_string_value==string][['user_id','restaurant_id']]
        interact_count = interact_count.groupby(['user_id','restaurant_id']).size().reset_index(name='is_positive')
        interact_count['is_positive'] = [True if c >= min_count else np.nan for c in interact_count['is_positive']]
        return interact_count.dropna()

    if string=='view_item':
    
        interact_count = simple_data[simple_data.event_name==string][['user_id','restaurant_id']]
        interact_count = interact_count.groupby(['user_id','restaurant_id']).size().reset_index(name='is_positive')
        interact_count['is_positive'] = [True if c >= min_count else np.nan for c in interact_count['is_positive']]
        return interact_count.dropna()    

def add_features_bin(score_data, simple_data, feat_to_add, min_count_feat):
    '''add all the feature interaction to score data binary'''        
    
    list_of_df = []
    
    for feat in feat_to_add:
        list_of_df.append(features_interaction(simple_data, feat, min_count_feat))
    
    df_features = pd.concat(list_of_df).reset_index(drop=True).drop_duplicates()
    
    usr_to_check = set(score_data.user_id).intersection(set(df_features.user_id))
    df_features = df_features.set_index('user_id').loc[list(usr_to_check),:].reset_index()
    rst_to_check = set(score_data.restaurant_id).intersection(set(df_features.restaurant_id))
    df_features = df_features.set_index('restaurant_id').loc[list(rst_to_check),:].reset_index()
    return pd.concat([score_data,df_features]).drop_duplicates().reset_index(drop=True)

def SimEuclid(User_Item_data, user1, user2, min_common_items=1):
    """Returns a distance-based similarity score for person1 and person2"""
    #Common rests between users
    common_cols = User_Item_data.columns[(User_Item_data.loc[user1,:] != 0) & (User_Item_data.loc[user2,:] != 0)]
    #Evaluating the two rating vectors
    vec1, vec2 = User_Item_data.loc[user1, common_cols], User_Item_data.loc[user2, common_cols]
        
    if(len(vec1) < min_common_items):
        return 0
    return 1.0/(1.0+euclidean(vec1, vec2))

def SimPearson(User_Item_data, user1, user2, min_common_items=1):
    """Returns a distance-based similarity score for person1 and person2"""
    #Common rests between users
    common_cols = User_Item_data.columns[(User_Item_data.loc[user1,:] != 0) & (User_Item_data.loc[user2,:] != 0)]
    #Evaluating the two rating vectors
    vec1, vec2 = User_Item_data.loc[user1, common_cols], User_Item_data.loc[user2, common_cols]
    if len(vec1) < min_common_items or len(vec1) < 2:
        return 0    
    res = pearsonr(vec1, vec2)[0]
    if np.isnan(res) or res < 0:
        return 0
    return res

def SimCosine(User_Item_data, user1, user2, min_common_items=1):
    """Returns a distance-based similarity score for person1 and person2"""
    #Common rests between users
    common_cols = User_Item_data.columns[(User_Item_data.loc[user1,:] != 0) & (User_Item_data.loc[user2,:] != 0)]
    #Evaluating the two rating vectors
    vec1, vec2 = User_Item_data.loc[user1, common_cols], User_Item_data.loc[user2, common_cols]
    vec1, vec2 = np.array([vec1.values]), np.array([vec2.values])
    if len(vec1[0]) < min_common_items:
        return 0    
    res = cosine_similarity(vec1, vec2)[0]
    if np.isnan(res) or res < 0:
        return 0
    return res[0]

#evaluating the similiraties considering the nan as zeros for fit_items2 method
def SimEuclid2(vec1, vec2):
    return 1.0/(1.0+euclidean(vec1, vec2))

#####################
#Collab Filter class#
#####################

class CollaborativeFiltering:
    """ Collaborative filtering using a custom sim(u,u'). """
    
    def __init__(self, DataFrame, similarity):
        """ Constructor """
        self.sim_method      = similarity# Gets recommendations for a person by using a weighted average
        self.df              = DataFrame
        self.sim_items_bin   = []
        self.sim             = {}
        self.sim_item        = {}
        self.user_mean       = []
        self.movie_mean      = []
        self.bin_neighbours  = []
        self.neighbours_item= []
     
    def fit_items_bin(self):
        """ Fit Items based model in binary form"""
        sparse_matrix       = sparse.csr_matrix(self.df)
        similarities        = self.sim_method(sparse_matrix.transpose())
        self.sim_items_bin  = pd.DataFrame(similarities, index=self.df.columns, columns = self.df.columns)
        
        self.bin_neighbours = pd.DataFrame(index=self.sim_items_bin.columns, columns=range(len(self.sim_items_bin.columns)))
        for i in self.sim_items_bin.columns:
            self.bin_neighbours.loc[i,:] = self.sim_items_bin.loc[0:,i].sort_values(ascending=False).index
    
    def fit(self):
        """ Prepare data structures for estimation. Similarity matrix for users """
        All_User_Index  = list(combinations(self.df.index, 2))
        
        for user1, user2 in All_User_Index:
            
            self.sim.setdefault(user1, {})
            self.sim.setdefault(user2, {})
    
            self.sim[user1][user2] = self.sim_method(self.df,user1,user2)
            self.sim[user2][user1] = self.sim[user1][user2]
            
    def fit_items(self):
        """ Fit Items based model """
        All_Restaurant_Index  = list(combinations(self.df.columns, 2))
        
        for rest1, rest2 in All_Restaurant_Index:
            
            self.sim_item.setdefault(rest1, {})
            self.sim_item.setdefault(rest2, {})
            self.sim_item[rest1][rest2] = self.sim_method(self.df.T,rest1,rest2)
            self.sim_item[rest2][rest1] = self.sim_item[rest1][rest2]
        
        sim_items_matrix     = pd.DataFrame(self.sim_item).fillna(1)
        
        self.neighbours_item = pd.DataFrame(index=sim_items_matrix.columns, columns=range(len(sim_items_matrix.columns)))
        for i in sim_items_matrix.columns:
            self.neighbours_item.loc[i,:] = sim_items_matrix.loc[:,i].sort_values(ascending=False).index

    def fit_items2(self, sim_function=SimEuclid2):
        '''evaluating the similiraties considering the nan as zeros adding N_neigh'''
        self.sim0 = pd.DataFrame(index=self.df.columns, columns=self.df.columns)
        for i,j in itertools.combinations(self.df, 2):
            v1=self.df.loc[:,i]
            v2=self.df.loc[:,j]
            self.sim0.loc[i,j]=sim_function(v1, v2)
            self.sim0.loc[j,i]=self.sim0.loc[i,j]
        self.sim0 = self.sim0.fillna(1)
        sim_matrix      = pd.DataFrame(self.sim0).fillna(1)
        self.neighbours_item = pd.DataFrame(index=sim_matrix.columns, columns=range(len(sim_matrix.columns)))
        for i in sim_matrix.columns:
            self.neighbours_item.loc[i,:] = sim_matrix.loc[:,i].sort_values(ascending=False).index
    

    def predict_K_item_bin(self, user, K, N_neigh):
        '''Predicting K recommended for binary model'''
        bin_neighbours = self.bin_neighbours.loc[:,:N_neigh]
        rests_liked = self.df.columns[self.df.loc[user,:]!=0]
        closest_to_check = set(bin_neighbours.loc[rests_liked,:].values.flatten())
        reduced_sim = self.sim_items_bin.loc[list(closest_to_check),list(closest_to_check)]
        user_vec = self.df.loc[user,list(closest_to_check)]
        score = reduced_sim.dot(user_vec).div(reduced_sim.sum(axis=1))
        score = score.drop(rests_liked)
        
        return score.nlargest(K)
    
    def predict_item(self, user_id, restaurant_id):
        """ Prediction made by Items based model """
        #Return 2 if nor movie nor user are not in the df
        if restaurant_id not in self.df.columns and user_id not in self.df.index:
            print('che caso!')
            return 2
        #Return the user mean if the restaurant is not in the df
        if restaurant_id not in self.df.columns:
            return self.df.loc[user_id,:][self.df.loc[user_id,:] != 0].mean()
        #Return the restaurant mean if the user is not in the df
        if user_id not in self.df.index:
            return self.df.loc[:,restaurant_id][self.df.loc[:,restaurant_id] != 0].mean() 
        #Alert if the restaurant has been already rated by the given user
        if self.df.loc[user_id, restaurant_id] != 0:
            print('User {} already rates this restaurant {}.'.format(user_id, restaurant_id))
            return
        
        sim_matrix      = pd.DataFrame(self.sim_item)
        df_transpose    = self.df.T     
        rest_to_check   = df_transpose.index[df_transpose.loc[:, user_id] != 0] 
        vec_rates       = df_transpose.loc[rest_to_check, user_id]
        vec_sim         = sim_matrix.loc[rest_to_check, restaurant_id]
        
        if (sum(vec_sim) == 0) and (vec_rates.mean() > 0): 
                # return the mean user rating if there is no similar item for the computation
            return vec_rates.mean()
        if (sum(vec_sim) == 0):
                # if the mean is negative or if user_id not in data set return mean item rating 
            return df_transpose.loc[restaurant_id,:][df_transpose.loc[restaurant_id,:] != 0].mean()
        
        return vec_rates.dot(vec_sim)/sum(vec_sim)        

    def predict_item3(self, user_id, restaurant_id, N_neigh):
        """ it not predicts just the >=4 and fills with ones the sim """
        #Return 2 if nor movie nor user are not in the df
        if restaurant_id not in self.df.columns and user_id not in self.df.index:
            print('che caso!')
            return 2
        #Return the user mean if the restaurant is not in the df
        if restaurant_id not in self.df.columns:
            return self.df.loc[user_id,:][self.df.loc[user_id,:] != 0].mean()
        #Return the restaurant mean if the user is not in the df
        if user_id not in self.df.index:
            return self.df.loc[:,restaurant_id][self.df.loc[:,restaurant_id] != 0].mean() 
        #Alert if the restaurant has been already rated by the given user
        if self.df.loc[user_id, restaurant_id] >= 4:
            print('User {} already likes this restaurant {}.'.format(user_id, restaurant_id))
            return
        
        sim_matrix      = pd.DataFrame(rec.sim0).fillna(1)
        df_transpose    = self.df.T
        rest_sim        = self.neighbours_item.loc[restaurant_id,1:N_neigh].values
        vec_rates       = df_transpose.loc[:, user_id].loc[rest_sim]
        vec_sim         = sim_matrix.loc[:, restaurant_id].loc[rest_sim]
        
        
        if (sum(vec_sim) == 0) and (vec_rates.mean() > 0): 
                # return the mean user rating if there is no similar item for the computation
            return vec_rates.mean()
        if (sum(vec_sim) == 0):
                # if the mean is negative or if user_id not in data set return mean item rating 
            return df_transpose.loc[restaurant_id,:][df_transpose.loc[restaurant_id,:] != 0].mean()
        
        return vec_rates.dot(vec_sim)/sum(vec_sim)   
    
            
    def predict(self, user_id, restaurant_id, verbose=False):
        """ Predicting based on the similarities of the users """
        #Return the user mean if the movie is not in the df
        if restaurant_id not in self.df.columns:
            return self.df.loc[user_id,:][self.df.loc[user_id,:] != 0].mean() 
        #Alert if the movie has been already rated by the given user
        if self.df.loc[user_id, restaurant_id] != 0:
            if verbose==True:
                print('User {} already rates movie {}.'.format(user_id, restaurant_id))
            return
        #Code based on rates X similarities dot product on the common subsets
        sim_matrix      = pd.DataFrame(self.sim)
        user_to_check   = self.df.index[self.df.loc[:, restaurant_id] != 0] 
        vec_rates       = self.df.loc[user_to_check, restaurant_id] 
        vec_sim         = sim_matrix.loc[user_to_check, user_id]
        
        if (sum(vec_sim) == 0) and (vec_rates.mean() > 0): 
                #return the mean movie rating if there is no similar for the computation
            return vec_rates.mean()
        if (sum(vec_sim) == 0):
                #if the mean is negative return mean user rating 
            return self.df.loc[user_id,:][self.df.loc[user_id,:] != 0].mean()
        
        return vec_rates.dot(vec_sim)/sum(vec_sim)


####################################################
#Function with differents metric to evaluate models#
####################################################

def compute_rmse(y_pred, y_true):
    """ Compute Root Mean Squared Error. """
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

def evaluate(estimation_func, UI_tr_data, UI_ts_data, score_df_test):
    """ RMSE-based predictive performance evaluation with pandas. """
    a = score_df_test.loc[:,'user_id'] ; b = score_df_test.index
    ids_to_estimate = list(zip(a, b))

    # we obtain the estimations. if a user u does not appear in the training set (not in the similarity
    # matrix, we assume an average rate of mean)
    mean = UI_tr_data.mean().mean()
    UI_tr_data = UI_tr_data.fillna(0) ; UI_ts_data = UI_ts_data.fillna(0)
    list_of_user = set(UI_tr_data.index)
    estimated = np.array([estimation_func(u,i) if u in list_of_user else mean for (u,i) in ids_to_estimate])
    real = [UI_ts_data.loc[u,i] for (u,i) in ids_to_estimate]
    
    return compute_rmse(estimated, real)

def prec_recall_k_rated(user_id, User_Items_train, test_df, predict_method, N_Niegh, K, verbose=False):
    """ Compute precision, recall and accuracy for the UI with scores for a user"""
    user_test_set = set(test_df.user_id)
    
    if user_id not in user_test_set:
        return np.nan
    
    real_liked = set(test_df[test_df.user_id==user_id].restaurant_id)
    
    rests_to_check = User_Items_train.columns[User_Items_train.loc[user_id,:]<4]
    sorted_rec = pd.Series({rest:predict_method(user_id, rest, N_Niegh) for rest in rests_to_check})\
                                                                               .sort_values(ascending=False)
    
    K_acc   = len(real_liked)
    top_K   = sorted_rec[:K]
    top_acc = sorted_rec[:K_acc]

    number_common_at_K = len(set(top_K.index).intersection(set(real_liked)))
    number_common_acc  = len(set(top_acc.index).intersection(set(real_liked)))

    if verbose==True:
        toprint=\
        '.....\nThe predicted restaurants are: {}\n.....\nThe real liked are: {}\n.....'.format(top_K\
                                                                                      ,list(real_liked))
        print(toprint)      
    
    accuracy       = number_common_acc/K_acc
    precision_at_K = number_common_at_K/K
    recall_at_K    = number_common_at_K/len(real_liked)

    return accuracy, precision_at_K, recall_at_K

def mean_prec_recall_at_k_rated(User_Items_train, test_df, predict_method, N_Niegh, K):
    """ Compute precision, recall and accuracy for the UI with scores meaning on all users"""    
    user_test_set = set(test_df.user_id)
    users_to_check = set(User_Items_train.index).intersection(user_test_set)
        
    scores = [prec_recall_k_rated(u, User_Items_train, test_df, predict_method, N_Niegh, K) \
                                                               for u in users_to_check]

    scores = [i for i in scores if pd.isnull(i) == False]
    mean_scores = [sum(y) / len(y) for y in zip(*scores)]
    mean_acc = mean_scores[0] ; mean_precision = mean_scores[1] ; mean_recall = mean_scores[2]
    User_Tested_Ratio = len(users_to_check)/len(User_Items_train)

    
    return mean_acc, mean_precision, mean_recall, User_Tested_Ratio

def rated_scores(df_init, model, min_liked_rest, min_liked_usr, model_similarity, N_Niegh, K):
    """ Compute precision, recall and accuracy over a train-test split for model with scores"""    
    
    train, test = train_test_split(data_init[['user_id','restaurant_id','is_positive']].dropna(), test_size=0.1)
    model.simple_data = data_init.drop(index=test.index)

    model.fit_data_score()
    model.fit_User_Items_Matrix(min_liked_rest, min_liked_usr)
    rec = CollaborativeFiltering(model.User_Items.fillna(0), model_similarity)
    rec.fit_items2()
    return mean_prec_recall_at_k_rated(model.User_Items.fillna(0), test, rec.predict_item3, N_Niegh, K)

def prec_recall_k_bin(user_id, User_Items_train, User_Items_test, predict_method, N_neigh, K, verbose=False):
    """ Compute precision, recall and accuracy for the binary UI for a given user"""    
    if user_id not in User_Items_test.index:
        return np.nan
    
    real_liked = User_Items_test.loc[user_id,:][User_Items_test.loc[user_id,:]!=0].index
    
    K_acc = len(real_liked)
    top_K_acc = predict_method(user_id, K_acc, N_neigh)
    top_K = predict_method(user_id, K, N_neigh)


    number_common_at_K_acc = len(set(top_K_acc.index).intersection(set(real_liked)))
    number_common_at_K     = len(set(top_K.index).intersection(set(real_liked)))
    
    accuracy = number_common_at_K_acc/K_acc     
    precision_at_K = number_common_at_K/K
    recall_at_K = number_common_at_K/len(real_liked)
    
    if precision_at_K ==0 and recall_at_K == 0: f1 = 0

    else: f1 = (2*precision_at_K*recall_at_K)/(precision_at_K+recall_at_K)

    if verbose==True:
        toprint=\
        '.....\nThe predicted restaurants are: {}\n.....\nThe real liked are: {}\n.....'.format(top_K\
                                                                                      ,list(real_liked))
        print(toprint)  
        
    return accuracy, f1, precision_at_K, recall_at_K


def mean_prec_recall_at_k_bin(User_Items_train, User_Items_test, predict_method, N_neigh, K):
    """ Compute precision, recall and accuracy for the binary UI meaning over all users"""        
    users_to_check = set(User_Items_train.index).intersection(set(User_Items_test.index))
    scores = [prec_recall_k_bin(u, User_Items_train, User_Items_test, predict_method, N_neigh, K) \
                                                               for u in users_to_check]

    scores = [i for i in scores if pd.isnull(i) == False]
    mean_scores = [sum(y) / len(y) for y in zip(*scores)]
    mean_acc = mean_scores[0] ; mean_f1 = mean_scores[1] ; mean_precision = mean_scores[2] ; mean_recall = mean_scores[3]
    User_Tested_Ratio = len(users_to_check)/len(User_Items_train)
    
    return mean_acc, mean_f1, mean_precision, mean_recall, User_Tested_Ratio 

def scores_bin(score_data_bin, repeat, N_neigh, K, min_rated_restaurants, min_rated_users, ts_size=0.1):
    """ Compute precision, recall and accuracy for the binary UI meaning over N=repeat train-test splits"""            
    tot_scores=[]
    for i in range(repeat): 
        train, test = train_test_split(score_data_bin, test_size=ts_size)
        UI_tr = get_User_Items_bin(train, min_rated_restaurants, min_rated_users)
        UI_ts = get_User_Items_bin(test)
    
        rec_bin = CollaborativeFiltering(UI_tr, cosine_similarity)
        rec_bin.fit_items_bin()
        tot_scores.append(mean_prec_recall_at_k_bin(UI_tr, UI_ts, rec_bin.predict_K_item_bin, N_neigh, K))
    mean_scores = [sum(y) / len(y) for y in zip(*tot_scores)]
    sigma_acc = np.sqrt(sum([(i[0]-mean_scores[0])**2 for i in tot_scores])/5)
    
    return mean_scores, sigma_acc 


def scores_bin_feat(score_data_bin,simple_data, feat_to_add, N_neigh, K, repeat,\
                   min_count_feat, min_rated_restaurants=5, min_rated_users=5,test_size=0.1):
    """ Compute precision, recall and accuracy for the binary UI meaning over N=repeat train-test splits adding feats"""                
    tot_scores=[]
    for i in range(repeat): 
        train, test = train_test_split(score_data_bin, test_size=test_size)
        train_feat = add_features_bin(train, simple_data, feat_to_add, min_count_feat)

        UI_tr_feat = get_User_Items_bin(train_feat, min_rated_restaurants, min_rated_users)
        UI_ts = get_User_Items_bin(test, 1, 1)
        rec_bin_feat = CollaborativeFiltering(UI_tr_feat, cosine_similarity)
        rec_bin_feat.fit_items_bin()
        
        tot_scores.append(mean_prec_recall_at_k_bin(UI_tr_feat, UI_ts, rec_bin_feat.predict_K_item_bin, N_neigh, K))

    mean_scores = [sum(y) / len(y) for y in zip(*tot_scores)]
    sigma_acc = np.sqrt(sum([(i[0]-mean_scores[0])**2 for i in tot_scores])/5)
    
    return mean_scores, sigma_acc        
        

def liked_statistics(User_Items_test):
    """It gives statistics of test UI binary matrix"""                    
    def liked_counter(user_id, User_Items_test):
        real_liked = User_Items_test.loc[user_id,:][User_Items_test.loc[user_id,:]!=0].index
        return len(real_liked)
    
    return pd.Series([liked_counter(i, User_Items_test) for i in User_Items_test.index]).describe()    
