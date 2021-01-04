import numpy as np
import pandas as pd
import os.path
from random import randint
import time

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = '../data/movies.csv'
users_file = '../data/users.csv'
ratings_file = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file = '../data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

def predict_test(movies, users, ratings, predictions):
    return predict_latent_factors(movies, users, ratings, predictions)


def normalization(matrix):
    dataframe_mean = matrix.mean(axis = 0, skipna = True)
    # print(matrix.subtract(dataframe_mean, axis = 'rows'))
    return matrix.subtract(dataframe_mean, axis = 'rows'), dataframe_mean

#####
##
## POPULATE USER/MOVIE MATRIX
##
#####

def populate_user_movie_matrix(movies, ratings):
    movie_data = pd.merge(ratings, movies, on='movieID')
    user_movie_matrix = movie_data.pivot_table(index='movieID', columns='userID', values='rating')
    return user_movie_matrix

#####
##
## CREATE USER/USER SIMILARITY MATRIX
##
#####

def create_user_similarity_matrix(user_movie_matrix):
    user_similarity_matrix = user_movie_matrix.replace(np.nan, 0)
    user_similarity_matrix = np.corrcoef(user_similarity_matrix.values.T)
    user_similarity_matrix = pd.DataFrame(data=user_similarity_matrix[0:, 0:], index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    return user_similarity_matrix

#####
##
## COLLABORATIVE FILTERING
##
#####

# Get top N neighbors
# Scan to make sure the top neighbors also have rated the movie

def get_n_nearest_neighbour(user_user_similarity_matrix, n, user_id):
    # -1 to get the indexing right
    all_neighbours = user_user_similarity_matrix.iloc[[user_id - 1]]
    # convert pandas dataframe to numpy ndarray. Get first column of array
    # (it only has one anyway, because we are only taking a single row)
    # to make it a 1D array from a 2D array.
    all_neighbours_np_array = all_neighbours.to_numpy()[0]
    # get index of the top n items in the array
    closest_n_neighbours_index = all_neighbours_np_array.argsort()[::-1][:n+1]
    # add 1 to the indecies to get the corresponding user_id
    # (because indecies start from 0, while user_ids start from 1).
    closest_n_neighbours = [i + 1 for i in closest_n_neighbours_index]
    # cut out the user itself from the list
    closest_n_neighbours = closest_n_neighbours[1::]
    # return list of n most similar user_ids
    return closest_n_neighbours


# Select top N dataset by similarity threshold
def get_neighbours(user_mov_matrix, similarity_matrix, user_id, movie_id, user_mean_rating, n):

    find_neighbors = get_n_nearest_neighbour(similarity_matrix, n, user_id)

    neighbor_ratings = user_mov_matrix.loc[movie_id][find_neighbors].fillna(0)
    # print(neighbor_ratings)
    neighbor_similarity = similarity_matrix[user_id].loc[find_neighbors]

    score = np.dot(neighbor_similarity, neighbor_ratings) + user_mean_rating[user_id]

    return round(score, 4)

def predict_collaborative_filtering(movies, users, ratings, predictions):
    start_t = time.time()
    user_movie_matrix = populate_user_movie_matrix(movies, ratings)
    print('creating user_movie matrix took:', time.time() - start_t)

    user_movie_matrix, user_mean_rating = normalization(user_movie_matrix)
    print('normalizing user_movie matrix took:', time.time() - start_t)

    user_sim_matrix = create_user_similarity_matrix(user_movie_matrix)
    print('creating user_user similarity matrix took:', time.time() - start_t)
    print('creating matrices took:', time.time() - start_t)

    # predictions_short = predictions.head(10000)
    # print(predictions_short)

    predicted_ratings = predictions.apply(lambda row: get_neighbours(user_movie_matrix, user_sim_matrix, row['userID'],
                                                                     row['movieID'], user_mean_rating, 5), axis=1)
    print('making predictions took:', time.time() - start_t)

    result_ratings = pd.Series(predicted_ratings).to_numpy()
    print('num of generated scores:', len(result_ratings))
    print('num of predictions read:', len(predictions))
    # result_ids = pd.Series(range(1, len(predicted_ratings) + 1)).to_numpy()
    ratings_final = []
    for i in range(0, len(predictions)):
        ratings_final.append((i + 1, result_ratings[i]))
    print('length of ratings final:', len(ratings_final))
    print('submitting predictions:', time.time() - start_t)
    return ratings_final


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    user_movie_matrix = populate_user_movie_matrix(movies, ratings)
    user_movie_matrix_normalized, user_mean_rating = normalization(user_movie_matrix)

    return



#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####

## //!!\\ TO CHANGE by your prediction function
#predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)
predictions = predict_test(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)

# if __name__ == "__main__":
