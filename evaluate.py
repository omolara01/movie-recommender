# evaluate.py
# Measures how accurate the collaborative filtering recommender is.
# Method: hold out 20% of ratings, predict them from the other 80%, and
# compare against a baseline that always predicts the average rating.
# Run: python evaluate.py 
import numpy as np
import pandas as pd

# Load the MovieLens 100K ratings (tab-separated: user, movie, rating, timestamp)
ratings = pd.read_csv("u.data", sep="\t",
                      names=["userID", "movieID", "rating", "timestamp"],
                      usecols=[0, 1, 2])

# Hold out 20% of ratings the model never sees, so the score is honest.
# random_state=42 makes the split repeatable.
test = ratings.sample(frac=0.2, random_state=42)
train = ratings.drop(test.index)

# User x movie grid built from the training ratings only
matrix = train.pivot_table(index="userID", columns="movieID", values="rating")
user_means = matrix.mean(axis=1)
# Subtract each user's own average so harsh and generous raters are comparable
centered = matrix.sub(user_means, axis=0)
# How similar every pair of users is (Pearson), needing at least 5 shared movies
similarity = matrix.T.corr(method="pearson", min_periods=5)
global_mean = train["rating"].mean()

def predict(user, movie, k=30):
    # Unknown user or movie: fall back to the overall average
    if user not in similarity.index or movie not in centered.columns:
        return global_mean
    raters = centered[movie].dropna()
    sims = similarity.loc[user, raters.index].drop(user, errors="ignore").dropna()
    sims = sims[sims > 0]            # use only positively similar users
    if sims.empty:
        return user_means[user]
    top = sims.nlargest(k)           # the 30 most similar users who rated this movie
    pred = user_means[user] + (top * raters[top.index]).sum() / top.abs().sum()
    return float(np.clip(pred, 1, 5))  # ratings only go from 1 to 5

preds = np.array([predict(u, m) for u, m in zip(test.userID, test.movieID)])
actual = test["rating"].values

# RMSE = typical size of the prediction error, in rating points
print("Model RMSE:   ", round(float(np.sqrt(np.mean((actual - preds) ** 2))), 3))
print("Baseline RMSE:", round(float(np.sqrt(np.mean((actual - global_mean) ** 2))), 3))
