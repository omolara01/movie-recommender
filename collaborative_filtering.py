# Collaborative filtering: find users with similar movie preferences
import argparse
import numpy as np
import pandas as pd
from similarity_score import pearson_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to an input user')
    # type=int because user IDs in the dataset are numbers; without it the
    # lookup below compares text to numbers and never finds the user.
    parser.add_argument('--user', dest='user', type=int, required=True, help='Input user ID')
    return parser


def find_similar_users(dataset, user, num_users):
    if user not in dataset.userID.values:
        raise TypeError("Can't find " + str(user) + " in the dataset")

    # Score the input user against every other user using Pearson correlation
    scores = np.array([[x, pearson_score(dataset, user, x)]
                       for x in dataset['userID'].unique() if x != user])

    # argsort sorts ascending, so [::-1] reverses it to put the highest scores first
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    top_users = scores_sorted[:num_users]
    return scores[top_users]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # Plain filenames work as long as the data files sit in the same folder as this script
    ratings = pd.read_csv("u.data",
                          sep='\t',
                          names=['userID', 'movieID', 'rating', 'timestamp'])

    # Loading movie titles; latin-1 is needed for accented characters in titles
    movies = pd.read_csv("u.item",
                         sep='|',
                         encoding='latin-1',
                         usecols=[0, 1],
                         names=['movieID', 'title'])

    # Merging so each rating row also carries its movie title
    data = pd.merge(ratings, movies, on='movieID')

    print('\nUsers similar to ' + str(user) + ':\n')
    similar_users = find_similar_users(data, user, 3)
    print('User\t\t\tSimilarity score')
    print('-' * 41)
    for item in similar_users:
        print(int(item[0]), '\t\t', round(float(item[1]), 2))
