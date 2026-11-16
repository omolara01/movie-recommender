# Movie Recommender System

A user-based collaborative filtering recommender built in Python, using the
MovieLens 100K dataset (943 users, 1,682 movies, 100,000 ratings).

Originally built during my NIIT training; the evaluation script was added later.

## What it does
- `similarity_score.py`: computes Euclidean and Pearson similarity between two users.
  Pearson scores need at least 10 shared movies, to avoid chance perfect scores.
- `collaborative_filtering.py`: finds the users most similar to a given user.
- `evaluate.py`: predicts held-out ratings and measures accuracy with RMSE.

## Results (80/20 holdout split, random_state=42)
| Method | RMSE |
|---|---|
| User-based collaborative filtering (Pearson, 30 neighbours) | 0.946 |
| Baseline: mean rating | 1.124 |

## How to run
1. Download the dataset from https://grouplens.org/datasets/movielens/100k/
   and put `u.data` and `u.item` in the same folder as the scripts.
2. Install requirements: `pip install -r requirements.txt`
3. Find similar users: `python collaborative_filtering.py --user 5`
4. Evaluate: `python evaluate.py`

## Limitations
- Similarity is calculated with loops, so finding similar users is slow.
- Users with few shared movies have unreliable similarity scores.

## Data
MovieLens 100K, GroupLens Research, University of Minnesota.
