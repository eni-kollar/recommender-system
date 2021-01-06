import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATA MINING CLASS

#### IDENTIFICATION
NAME:
SURNAME:
STUDENT ID:
KAGGLE ID:


### NOTES
This files is an example of what your code should look like. 
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = '../data/movies.csv'
users_file = '../data/users.csv'
ratings_file = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file = '../data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])

def normalization(matrix):
    dataframe_mean = matrix.mean(axis = 0, skipna = True)
    # print(matrix.subtract(dataframe_mean, axis = 'rows'))
    return matrix.subtract(dataframe_mean, axis = 'rows'), dataframe_mean

#####
##
## POPULATE USER/MOVIE MATRIX
##
#####

def populate_user_movie_matrix(users, ratings, movies):

    user_data = pd.merge(ratings, users, on='userID')
    user_movie_matrix = user_data.pivot_table(index='userID', columns='movieID', values='rating')

    for i in range(1, len(movies)):
        if not i in user_movie_matrix.columns:
            user_movie_matrix[i] = np.nan
    user_movie_matrix = user_movie_matrix.reindex(sorted(user_movie_matrix.columns), axis=1)
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

def get_n_nearest_neighbour(user_user_similarity_matrix, n, movie_id, rating_matrix, user_id):
    # -1 to get the indexing right
    print('movieID', movie_id)
    all_neighbours = user_user_similarity_matrix.iloc[[movie_id]]
    # convert pandas dataframe to numpy ndarray. Get first column of array
    # (it only has one anyway, because we are only taking a single row)
    # to make it a 1D array from a 2D array.
    all_neighbours_np_array = all_neighbours.to_numpy()[0]
    # get index of the top n items in the array
    closest_n_neighbours_index = all_neighbours_np_array.argsort()[::-1][:n+1]
    # add 1 to the indecies to get the corresponding user_id
    # (because indecies start from 0, while user_ids start from 1).
    j = 0
    res = []
    for i in range(1, len(closest_n_neighbours_index)):
        if (j == 5) and (rating_matrix.iloc[closest_n_neighbours_index[i]][user_id] > 0):
            rating_matrix[i] = np.nan
            res.append(i)
            j = j + 1

    # closest_n_neighbours = [i + 1 for i in closest_n_neighbours_index]
    # cut out the user itself from the list
    closest_n_neighbours = res
    print(closest_n_neighbours)
    # return list of n most similar user_ids
    return closest_n_neighbours


# Select top N dataset by similarity threshold
def get_neighbours(user_mov_matrix, similarity_matrix, user_id, movie_id, movie_mean_rating, n):

    find_neighbors = get_n_nearest_neighbour(similarity_matrix, n, movie_id, user_mov_matrix, user_id)

    #  neighbor_ratings = user_mov_matrix.loc[movie_id][find_neighbors].fillna(0)
    #     # print(neighbor_ratings)
    #     neighbor_similarity = similarity_matrix[user_id].loc[find_neighbors]

    neighbor_ratings = user_mov_matrix.loc[user_id][find_neighbors].fillna(0)
    print(neighbor_ratings)
    neighbor_similarity = similarity_matrix[movie_id].loc[find_neighbors]



    score = np.dot(neighbor_similarity, neighbor_ratings) + movie_mean_rating[movie_id]
    # print(user_mean_rating)
    print('mean:', movie_mean_rating[movie_id])
    print('score:', score)
    return round(score, 4)

def predict_collaborative_filtering(movies, users, ratings, predictions):
    user_movie_matrix = populate_user_movie_matrix(users, ratings, movies)
    print(user_movie_matrix)

    user_movie_matrix, user_mean_rating = normalization(user_movie_matrix)

    user_sim_matrix = create_user_similarity_matrix(user_movie_matrix)

    print(user_sim_matrix)
    predictions = predictions.head(5)

    predicted_ratings = predictions.apply(lambda row: get_neighbours(user_movie_matrix, user_sim_matrix, row['userID'],
                                                                     row['movieID'], user_mean_rating, 5), axis=1)

    print('all predictions: ', predicted_ratings)

    result_ratings = pd.Series(predicted_ratings).to_numpy()

    ratings_final = []
    for i in range(0, len(predictions)):
        ratings_final.append((i + 1, result_ratings[i]))
    return ratings_final




def predict(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)