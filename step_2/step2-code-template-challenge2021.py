import numpy as np
import pandas as pd
from random import randint
# import time


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
    return matrix.subtract(dataframe_mean, axis = 'columns'), dataframe_mean

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

def create_cos_similarity_matrix(user_movie_matrix):

    data = user_movie_matrix.replace(np.nan, 0)
    data = np.dot(data.T, data)/np.linalg.norm(data)/np.linalg.norm(data)
    user_similarity_matrix = pd.DataFrame(data=data, index=user_movie_matrix.columns,
                                          columns=user_movie_matrix.columns)
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
    all_neighbours = user_user_similarity_matrix.iloc[[movie_id - 1]]
    # convert pandas dataframe to numpy ndarray. Get first column of array
    # (it only has one anyway, because we are only taking a single row)
    # to make it a 1D array from a 2D array.
    all_neighbours_np_array = all_neighbours.to_numpy()[0]
    # get index of the top n items in the array
    closest_n_neighbours_index = all_neighbours_np_array.argsort()[::-1][:n+1]

    closest_n_neighbours = [i + 1 for i in closest_n_neighbours_index]
    closest_n_neighbours = closest_n_neighbours[1:]

    return closest_n_neighbours


# Select top N dataset by similarity threshold
def get_neighbours(user_mov_matrix, similarity_matrix, user_id, movie_id, movie_mean_rating, n):

    find_neighbors = get_n_nearest_neighbour(similarity_matrix, n, movie_id, user_mov_matrix, user_id)


    neighbor_ratings = user_mov_matrix.iloc[user_id - 1][find_neighbors].fillna(0)
    # print('n_r:', neighbor_ratings)
    neighbor_similarity = similarity_matrix.iloc[movie_id - 1][find_neighbors]
    # print('n_s:', neighbor_similarity)



    # wrk out for cases where there are no ratings - use average rating from that user
    # 1. filter out neighbours who have not rated the movie
    # 2. DONE! add it so if there are no neighbours (if the movie has no ratings), return average rating for that user_id

    score = np.dot(neighbor_similarity, neighbor_ratings)/np.sum(neighbor_similarity) + movie_mean_rating[movie_id]
    # print(user_mean_rating)
    # print('mean:', movie_mean_rating[movie_id])
    # print('score:', score)

    if np.isnan(score):
        score = get_av_user_rating(user_id, user_mov_matrix)
        # score = 2.5
    return round(score, 4)

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # start_t = time.time()
    user_movie_matrix_b4 = populate_user_movie_matrix(users, ratings, movies)
    # print(user_movie_matrix)
    # print(user_movie_matrix)
    user_movie_matrix, user_mean_rating = normalization(user_movie_matrix_b4)
    print(user_movie_matrix)

    user_sim_matrix = create_user_similarity_matrix(user_movie_matrix_b4)
    user_sim_matrix = user_sim_matrix.fillna(0)

    # print(user_sim_matrix)
    # predictions = predictions.head(5)

    predicted_ratings = predictions.apply(lambda row: get_neighbours(user_movie_matrix, user_sim_matrix, row['userID'],
                                                                     row['movieID'], user_mean_rating, 5), axis=1)

    # print('all predictions: ', predicted_ratings)

    result_ratings = pd.Series(predicted_ratings).to_numpy()

    ratings_final = []
    for i in range(0, len(predictions)):
        ratings_final.append((i + 1, result_ratings[i]))
    # print('submitting predictions:', time.time() - start_t)
    return ratings_final

def get_cf_predicted_ratings(movies, users, ratings, predictions):
    user_movie_matrix_b4 = populate_user_movie_matrix(users, ratings, movies)

    user_movie_matrix, user_mean_rating = normalization(user_movie_matrix_b4)
    user_sim_matrix = create_user_similarity_matrix(user_movie_matrix_b4)
    user_sim_matrix = user_sim_matrix.fillna(0)

    predicted_ratings = predictions.apply(lambda row: get_neighbours(user_movie_matrix, user_sim_matrix, row['userID'],
                                                                     row['movieID'], user_mean_rating, 10), axis=1)
    result_ratings = pd.Series(predicted_ratings).to_numpy()
    return result_ratings



def get_av_user_rating(user_id, user_movie_matrix):
    no_zeros_matrix = user_movie_matrix.replace(0, np.NaN)
    user_mean = no_zeros_matrix.mean(axis=1, skipna=True)
    return user_mean[user_id - 1]


#########################
#########################




def predict_latent_factors_s1(movies, users, ratings, predictions):

    user_data = pd.merge(users, ratings, on='userID')
    user_movie_matrix = user_data.pivot_table(index='userID', columns='movieID', values='rating')
    # result_df = pd.DataFrame(movies.apply(lambda x: users.apply(lambda y: return_value(x, y, user_movie_matrix, df))))

    print(user_movie_matrix)
    # d = pd.DataFrame(0, index=np.arange(movies), columns=movies.columns)

    numpy_user_movie_matrix = user_movie_matrix.as_matrix()

    # user_ratings_mean = np.mean(numpy_user_movie_matrix, axis=1, skipna=True)
    user_ratings_mean = np.nanmean(numpy_user_movie_matrix, axis=1)
    # numpy_user_movie_matrix.fillna(0)

    user_movie_matrix_normalized = numpy_user_movie_matrix - user_ratings_mean.reshape(-1, 1)
    temp = pd.DataFrame(user_movie_matrix_normalized)
    temp = temp.fillna(0)
    user_movie_matrix_normalized = temp.as_matrix()

    U, sigma, V_t = np.linalg.svd(user_movie_matrix_normalized, full_matrices=False)
    k = 25

    sigma = np.diag(sigma[:k])

    all_user_predicted_ratings = np.matmul(U[:, :k], sigma)
    all_user_predicted_ratings = np.matmul(all_user_predicted_ratings, V_t[:k]) + user_ratings_mean.reshape(-1, 1)

    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_movie_matrix.columns.astype(str))

    predicted_ratings = predictions.apply(lambda row: predictions_lf(preds_df, row['userID'],
                                                                      row['movieID'], user_ratings_mean), axis=1)
    result_ratings = pd.Series(predicted_ratings).to_numpy()

    ############
    # GET CF RATINGS
    cf_weight = 0.60
    lf_weight = 0.40
    cf_ratings = get_cf_predicted_ratings(movies, users, ratings, predictions)
    ratings_final = []
    for i in range(0, len(predictions)):
        ratings_final.append((i + 1, result_ratings[i]*lf_weight + cf_ratings[i]*cf_weight))



    #
    # for i in range(0, len(ratings_final)):
    #     ratings_final[i] = result_ratings[i]*lf_weight + cf_ratings[i]*cf_weight

    return ratings_final

def predictions_lf(predictions_df, user_id, movie_id, means):
    try:
        ind = predictions_df.columns.get_loc(str(movie_id))
        return np.round(predictions_df.iloc[user_id - 1, ind],4)
    except:
        return np.round(means[user_id-1], 4)






##########################
##########################

def cal_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost




def stochastic_gradient_descent(P, Q, user_movie_matrix):

    no_of_latent_features_k = 20
    maximum_no_of_iterations = 50
    learning_rate = 0.005
    regularization_lambda = 0.1
    cost = 0.0

    for i in range(maximum_no_of_iterations):
        # calculate RMSE

        predicted_values = np.matmul(P, Q)
        actual_values = user_movie_matrix

        P_num_rows, P_num_cols = P.shape
        Q_num_rows, Q_num_cols = Q.shape

        for i in range(0, len(P[:])):
            for j in range(len(P[i,:])):
                row = P[i, :]
                column = P[:, j]
                predicted_value = P[i][j]
                actual_value = actual_values[i][j]
                error_rate_P = sum_of_squares_error(predicted_value, actual_value)
                e_x_i = (actual_value - predicted_value)
                P[i] = P[i] + learning_rate*(error_rate_P*Q[j] - regularization_lambda*P[i])


        # calculate RMSE after a few iterations and see how it is

        for row in Q.rows:
            for column in Q.columns:
                predicted_value = Q[row][column]
                actual_value = actual_values[row][column]
                error_rate_Q = sum_of_squares_error(predicted_value, actual_value)
                # error_rate_P = sum_of_squares_error(predicted_value, actual_values)
                Q[column] = P[column] + learning_rate*(error_rate_Q*P[row] - regularization_lambda*Q[column])


    error_rate = root_mean_square_error(predicted_values, actual_values)
    all_predicted_ratings = P * Q

def sum_of_squares_error(predicted_values, actual_values):
    sse = np.sum((predicted_values - actual_values) ** 2)
    return sse

def root_mean_square_error(estimated_values, actual_values):
    # get root mean square error or predicted vs. actual movie ratings
    mse_sum = 0
    for i in range(0, len(estimated_values)):
        rmse_sum = pow((estimated_values[i] - actual_values[i]), 2)
    mse = mse_sum/len(estimated_values)
    rmse = np.sqrt(mse)
    return rmse

def predict_latent_factors(movies, users, ratings, predictions):

    user_data = pd.merge(users, ratings, on='userID')
    user_movie_matrix = user_data.pivot_table(index='userID', columns='movieID', values='rating')

    numpy_user_movie_matrix = user_movie_matrix.as_matrix()

    #################
    # create P and Q
    # P users (rows) * factors
    # Q  factors * movies (columns)
    P = np.full((len(users), 20), 0.001)
    Q = np.full((20, len(movies)), 0.001)

    predicted_ratings = stochastic_gradient_descent(P, Q, numpy_user_movie_matrix)
    result_ratings = pd.Series(predicted_ratings).to_numpy()

    ratings_final = []
    for i in range(0, len(predictions)):
        ratings_final.append((i + 1, result_ratings[i]))

    return ratings_final

# def predictions_lf(predictions_df, user_id, movie_id, means):
#     try:
#         ind = predictions_df.columns.get_loc(str(movie_id))
#         return np.round(predictions_df.iloc[user_id - 1, ind],4)
#     except:
#         return np.round(means[user_id-1], 4)

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_latent_factors_s1(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)