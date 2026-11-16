import pandas as pd
import argparse
#import json
import numpy as np
import os
#lloading ratings

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description= 'Compute Similarity Score')
    parser.add_argument('--user1', dest= 'user1', type=int,
                       required= True, help= 'First User')
    parser.add_argument('--user2', dest= 'user2', type=int,
                       required= True, help= 'Second User')
    parser.add_argument('--score-type', dest= 'score_type',
                       required= True, 
                       choices=['Euclidean','Pearson'],
                       help= 'similarity metric to use')
    return parser 

# To calculate the similarity score between users

# compute the Euclidean distance score between user1 and user2
def euclidean_score(data, user1, user2):
    if user1 not in data.userID.values:
        raise TypeError('Cannot find ' + str(user1) + ' in the data')
            
    if user2 not in data.userID.values:
        raise TypeError('Cannot find ' + str(user2) + ' in the data')
        

# Movies rated by both user1 and user2
    common_movies = {}

# Extract the movies rated by both users; note ours is a dataframe:

    for item in data[data['userID'] == user1]['movieID'].values:
        if item in data[data['userID'] == user2]['movieID'].values:
            common_movies[item] = 1
        
# If there are no common movies between the users, then the score is 0

    if len(common_movies) == 0:
        return 0

# compute the squared differences between the ratings and use it to compute the Euclidean score:

    squared_diff = []

    for item in data[data['userID'] == user1]['movieID'].values:
        if item in data[data['userID'] == user2]['movieID'].values:
            squared_diff.append(np.square(data.loc[(data.userID == user1) & (data.movieID == item)]['rating'].iloc[0] - 
                                          data.loc[(data.userID == user2) & (data.movieID == item)]['rating'].iloc[0]))
    return 1 / (1 + np.sqrt(np.sum(squared_diff))) # Euclidean distance formular

# compute the Pearson correlation score between user1  and user2

def pearson_score(data, user1, user2, min_common=10):
    if user1 not in data.userID.values:
        raise TypeError('Cannot find ' + str(user1) + ' in the data')
    if user2 not in data.userID.values:
        raise TypeError('Cannot find ' + str(user2) + ' in the data')

# Movies rated by both user1 and user2
    common_movies = {}

# Extract the movies rated by both users:

    for item in data[data['userID'] == user1]['movieID'].values:
        if item in data[data['userID'] == user2]['movieID'].values:
            common_movies[item] = 1
        
    num_ratings = len(common_movies)

# if there are no common movies between user1 and user2, then the score is 0
      # Too few shared movies makes a correlation of 1.0 happen by chance,
    # so such pairs are treated as having no reliable similarity.
    if num_ratings < min_common:
        return 0

# calculate the sum of ratings of all the common movies 
    user1_sum = np.sum([data.loc[(data['userID'] == user1) & (data.movieID == item)]['rating'].iloc[0]
                        for item in common_movies])
    user2_sum = np.sum([data.loc[(data['userID'] == user2) & (data.movieID == item)]['rating'].iloc[0]
                        for item in common_movies])
    
# calculate the sum of squares of ratings of all the common movies     
    user1_squared_sum = np.sum([np.square(data.loc[(data['userID'] == user1) & (data.movieID == item)]['rating'].iloc[0])
                                           for item in common_movies])
    user2_squared_sum = np.sum([np.square(data.loc[(data['userID'] == user2) & (data.movieID == item)]['rating'].iloc[0])
                                           for item in common_movies])
    
# calculate the sum of products of the ratings of the common movies    
    sum_of_products = np.sum([data.loc[(data['userID'] == user1) & (data.movieID == item)]['rating'].iloc[0]
                              * 
                              data.loc[(data['userID'] == user2) & (data.movieID == item)]['rating'].iloc[0]
                              for item in common_movies])
    
# calculate the pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
# if there is no deviation, then the score is 0:    
    if Sxx * Syy == 0:
        return 0
# Return the paerson score:    
    return Sxy / np.sqrt(Sxx * Syy)
    
# Define the main function and parse the input arquments

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    user2 = args.user2
    score_type = args.score_type

    # Plain filenames work as long as the data files sit in the same folder as this script
    ratings = pd.read_csv("u.data",
                          sep='\t',
                          names=['userID', 'movieID', 'rating', 'timestamp'])

    # Loading movie info; latin-1 is needed for accented characters in titles
    movies = pd.read_csv("u.item",
                         sep='|',
                         encoding='latin-1',
                         usecols=[0, 1],
                         names=['movieID', 'title'])

    # Merging
    data = pd.merge(ratings, movies, on='movieID')

    # Compute the similarity score based on the input arguments
    if score_type == 'Euclidean':
        print("\nEuclidean score:")
        print(euclidean_score(data, user1, user2))
    else:
        print("\nPearson score:")
        print(pearson_score(data, user1, user2))
