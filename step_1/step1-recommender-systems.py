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


def create_user_movie_matrix(movies, users, ratings):
    # users are rows, movies are columns
    user_movie_matrix = np.empty((users.size + 1, movies.size + 1))
    for i in range(0, 1000):  # len(ratings)
        user = ratings.iloc[[i]]['userID']
        movie = ratings.iloc[[i]]['movieID']
        rating = ratings.iloc[[i]]['rating']
        user_movie_matrix[user, movie] = rating
    print("finished")
    print(user_movie_matrix)
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
    user_user_matrix = np.empty((users.size, users.size))
    for user1 in range(0, len(user_movie_matrix)):
        for user2 in range(0, len(user_movie_matrix)):
            user1_ratings = pd.Series(user_movie_matrix[user1])
            user2_ratings = pd.Series(user_movie_matrix[user2])
            correlation_coefficient = user1_ratings.corr(user2_ratings, method="pearson")
            if not pd.isnull(correlation_coefficient):
                user_user_matrix[user1][user2] = correlation_coefficient
                print("correlation coefficient", correlation_coefficient)
    print("user user matrix:")
    print(user_user_matrix)
    return user_user_matrix

def predict_collaborative_filtering(movies, users, ratings, predictions):

    # Method 1: Collaborative Filtering
    # 1. First create an user/movie matrix containing all of the ratings.
    user_movie_matrix = create_user_movie_matrix(movies, users, ratings)
    # 2. With this matrix you can compute the utility matrix containing the similarities between the users
    # Use Jaccard similarity  J(A, B) = (A intersect B) / (A union B) = (A intersect B) / (A + B - (A intersect B))
    user_user_similarity_matrix = pearson_correlation_coefficient(user_movie_matrix, users)
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
predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = "o"
    # predictions = [map(str, row) for row in predictions]
    # predictions = [','.join(row) for row in predictions]
    # predictions = 'Id,Rating\n'+'\n'.join(predictions)

    #Writes it dowmn
    submission_writer.write(predictions)
