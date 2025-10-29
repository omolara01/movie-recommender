#collaborative filtering to get users with similiar references
import argparse
import numpy as np
import pandas as pd
import os
from similarity_score import pearson_score
def build_arg_parser():
    parser=argparse.ArgumentParser(description = 'Find users who are similar to input user')
    parser.add_argument('--user',dest='user',required = True, help= 'Input user')
    return parser

def find_similar_users(dataset,user,num_users):
    if user not in dataset.userID.values:
        raise TypeError ('cant find '+str(user)+ 'in the dataset')
    scores = np.array([[x, pearson_score(dataset ,user ,x)]for x in dataset['userID'].unique() if x !=user])
    
     # to check the count of users that the code has worked for :
    '''l,i=[],0
    for x in dataset['userID'].unique():
            if x !=user:
                l.append([x,pearson_score(dataset,user,x)])
                i+=1
                print('count:',i)
            score = np.array(l)'''
            
    scores_sorted = np.argsort(scores[:,1])[::-1]
    
    top_users = scores_sorted[:num_users]
    return scores[ top_users]


if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    
    ratings = pd.read_csv(os.path.abspath("movie-recommender-app\\u.data"),
                    sep= '\t',
                    names= ['userID', 'movieID', 'rating', 'timestamp'])

 #Loading movie info
    movies =pd.read_csv(os.path.abspath("movie-recommender-app\\u.item"),
                    sep= '|',
                    encoding = 'latin-1',
                    usecols= [0,1],
                    names= ['movieID', 'title'])

    #merging
    data=pd.merge(ratings,movies,on='movieID')
    
    print('\nUsers similar to '+str(user)+':\n')
    similar_users = find_similiar_users(data,user,3)
    print ('User\t\t\tSimilarity score')
    print ('-' * 41)
    for item in similar_users:
        print(item[0], '\t\t',round(float(item[1]),2))
