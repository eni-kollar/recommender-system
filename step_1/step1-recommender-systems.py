import numpy as np
import pandas as pd
import os.path
from random import randint

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

#Where data is located
movies_file = '../data/movies.csv'
users_file = '../data/users.csv'
ratings_file = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file = '../data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

# Apply collaborative filtering
#####
##
## Step 1: First create an user/movie matrix containing all of the ratings.
##
#####


# def create_user_movie_matrix(movies, users, ratings):
#     # users are rows, movies are columns
#     user_movie_matrix = np.empty((users.size + 1, movies.size + 1))
#     for i in range(0, 1000):  # len(ratings)
#         user = ratings.iloc[[i]]['userID']
#         movie = ratings.iloc[[i]]['movieID']
#         rating = ratings.iloc[[i]]['rating']
#         user_movie_matrix[user, movie] = rating
#     print("finished")
#     print(user_movie_matrix)
#     return user_movie_matrix


def create_user_movie_matrix(movies, ratings):
    movie_data = pd.merge(ratings, movies, on='movieID')
    user_movie_matrix = movie_data.pivot_table(index='movieID', columns='userID', values='rating')
    return user_movie_matrix

    #####
##
## COLLABORATIVE FILTERING
##
#####

# def jaccard_similarity(user_movie_matrix, users):
#     # compute how similar each user is to every other user
#     # mirror other side of the matrix to improve time complexity
#     user_user_matrix = np.empty((users.size, users.size))
#     for i in range(0, len(user_movie_matrix)):
#         print("second_loop", i)
#         break
#         user1 =
    # def jaccard_similarity(x, y):
    #     intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    #     union_cardinality = len(set.union(*[set(x), set(y)]))
    #     return intersection_cardinality / float(union_cardinality)


def pearson_correlation_coefficient(user_movie_matrix, users):
    user_similarity_matrix = user_movie_matrix.replace(np.nan, 0)
    user_similarity_matrix = np.corrcoef(user_similarity_matrix.values.T)
    print(type(user_similarity_matrix))
    user_similarity_df = pd.DataFrame(data=user_similarity_matrix[0:, 0:], index=user_movie_matrix.columns,
                                          columns=user_movie_matrix.columns)
    # user_similarity_df = pd.DataFrame(user_similarity_matrix, columns=user_similarity_matrix.columns)
    # user_similarity_matrix = pd.DataFrame(index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    # # print(user_similarity_matrix.iloc[[1]])
    # # print("Users index:", len(users.index))
    # for user1 in range(0, len(users.index)):
    #     user1_ratings = user_movie_matrix.iloc[[user1]]
    #     # print(len(user_movie_matrix.index))
    #     # print("check1")
    #     # print(user_movie_matrix.iloc[[3695]])
    #     # print("check2")
    #     for user2_ratings in user_movie_matrix.iterrows():
    #     # for user2 in range(user1, len(users.index)):
    #     #     print("user1:", user1, " user2:", user2)
    #     #     # user1_ratings = user_movie_matrix.iloc[[user1]]
    #     #     user2_ratings = user_movie_matrix.iloc[[user2]]
    #     #     print("check2")
    #         correlation_coefficient = user1_ratings.corr(user2_ratings, method='pearson')  #method='pearson'
    #     #     correlation_coefficient = 6
    #         user_similarity_matrix.iloc[user1][user2] = correlation_coefficient
    #         user_similarity_matrix.iloc[user2][user1] = correlation_coefficient
    #         print("check3")
    #     print("new outer loop:", user1)
    return user_similarity_df


    # user_user_matrix = np.empty((len(users.index), len(users.index)))
    #         user1_ratings = pd.Series(user_movie_matrix[user1])
    #         user2_ratings = pd.Series(user_movie_matrix[user2])
    # #
    # #         # print(correlation_coefficient)
    # #         if not pd.isnull(correlation_coefficient):
    # # #         if not pd.isnull(correlation_coefficient):
    # # #             user_user_matrix[user1][user2] = correlation_coefficient
    # # #             print("correlation coefficient", correlation_coefficient)
    # print("user user matrix:")
    # print(user_similarity_matrix)
def get_n_nearest_neighbour(user_user_similarity_matrix, n, user_id):
    # -1 to get the indexing right
    all_neighbours = user_user_similarity_matrix.iloc[[user_id - 1]]
    # convert pandas dataframe to numpy ndarray. Get first column of array
    # (it only has one anyway, because we are only taking a single row)
    # to make it a 1D array from a 2D array.
    all_neighbours_np_array = all_neighbours.to_numpy()[0]
    # get index of the top n items in the array
    closest_n_neighbours_index = all_neighbours_np_array.argsort()[::-1][:n]
    # add 1 to the indecies to get the corresponding user_id
    # (because indecies start from 0, while user_ids start from 1)
    closest_n_neighbours = [i + 1 for i in closest_n_neighbours_index]
    # return list of n most similar user_ids
    return closest_n_neighbours


def predict_collaborative_filtering(movies, users, ratings, predictions):

    # Method 1: Collaborative Filtering
    # 1. First create an user/movie matrix containing all of the ratings.
    user_movie_matrix = create_user_movie_matrix(movies, ratings)
    print(user_movie_matrix)
    # 2. With this matrix you can compute the utility matrix containing the similarities between the users
    # Use Jaccard similarity  J(A, B) = (A intersect B) / (A union B) = (A intersect B) / (A + B - (A intersect B))
    user_user_similarity_matrix = pearson_correlation_coefficient(user_movie_matrix, users)
    print(user_user_similarity_matrix)
    # 3. You can use these similarities to predict the ratings given in the Predictions file(Predictions.csv).
    get_n_nearest_neighbour(user_user_similarity_matrix, 5, 4)
    pass


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    pass



#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####

#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####

## //!!\\ TO CHANGE by your prediction function
# predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)
movies_df = pd.DataFrame(movies_description)
users_df = pd.DataFrame(users_description)
ratings_df = pd.DataFrame(ratings_description)
predictions_df = pd.DataFrame(predictions_description)

print("converted to pd datafrmes")
# predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)
predictions = predict_collaborative_filtering(movies_df, users_df, ratings_df, predictions_df)
#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = "o"
    # predictions = [map(str, row) for row in predictions]
    # predictions = [','.join(row) for row in predictions]
    # predictions = 'Id,Rating\n'+'\n'.join(predictions)

    #Writes it dowmn
    submission_writer.write(predictions)
