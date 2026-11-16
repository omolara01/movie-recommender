# Recommends movies to a user, based on what similar users rated highly.
import argparse
import numpy as np
import pandas as pd
from similarity_score import pearson_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Recommend movies to an input user')
    # type=int: user IDs in the dataset are numbers, so text input would never match
    parser.add_argument('--user', dest='user', type=int, required=True, help='Input user ID')
    return parser


def get_recommendations(dataset, input_user):
    if input_user not in dataset.userID.values:
        raise TypeError('Cannot find ' + str(input_user) + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}
    seen = set(dataset[dataset['userID'] == input_user].movieID.values)

    for user in [x for x in dataset.userID.unique() if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)
        # Only learn from users who are clearly similar
        if similarity_score < 0.8:
            continue

        user_ratings = dataset[dataset['userID'] == user]
        for movie_id, rating in zip(user_ratings.movieID, user_ratings.rating):
            if movie_id in seen:        # skip movies the input user has already rated
                continue
            overall_scores[movie_id] = rating * similarity_score
            similarity_scores[movie_id] = similarity_score

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Normalising by the similarity score gives each movie a comparable rating, best first
    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()])
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
    return [int(movie_id) for _, movie_id in movie_scores]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings = pd.read_csv("u.data", sep='\t',
                          names=['userID', 'movieID', 'rating', 'timestamp'])
    movies = pd.read_csv("u.item", sep='|', encoding='latin-1',
                         usecols=[0, 1], names=['movieID', 'title'])
    data = pd.merge(ratings, movies, on='movieID')

    print('\nMovie recommendations for ' + str(user) + ':')
    recommendations = get_recommendations(data, user)
    for i, movie_id in enumerate(recommendations):
        if isinstance(movie_id, str):   # the "No recommendations possible" message
            print(movie_id)
            break
        title = data.loc[data['movieID'] == movie_id]['title'].iloc[0]
        print(str(i + 1) + '. ' + title)
