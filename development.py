# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading csv file into pandas dataframe
# read ratings file
ratings = pd.read_csv('ratings.csv')

# read properties data file
property_info = pd.read_csv('propertiesprocessed.csv')
property_info.set_index('property_id', inplace=True)

#split data
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(ratings, test_size = 0.30, random_state = 42)

# Specify file names for the training and testing sets
train_file = 'train_data.csv'
test_file = 'test_data.csv'

# Save training and testing sets to separate CSV files
X_train.to_csv(train_file, index=False)
X_test.to_csv(test_file, index=False)

#print(X_train.shape)
#print(X_test.shape)

# pivot ratings into movie features
user_data = X_train.pivot(index = 'userId', columns = 'propertyId', values = 'rating').fillna(0)
user_data.head()

# make a copy of train and test datasets
dummy_train = X_train.copy()
dummy_test = X_test.copy()

dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)

#The properties not rated by user is marked as 1 for prediction 
dummy_train = dummy_train.pivot(index = 'userId', columns = 'propertyId', values = 'rating').fillna(1)

# The properties not rated by user is marked as 0 for evaluation 
dummy_test = dummy_test.pivot(index ='userId', columns = 'propertyId', values = 'rating').fillna(0)


#ITEM BASED COLLABORATIVE FILTERING MODEL

#set property features
property_features = X_train.pivot(index = 'propertyId', columns = 'userId', values = 'rating').fillna(0)
#property_features.head()

from sklearn.metrics.pairwise import cosine_similarity

# Item Similarity Matrix using Cosine similarity as a similarity measure between Items
item_similarity = cosine_similarity(property_features)
item_similarity[np.isnan(item_similarity)] = 0
#print(item_similarity)
#print("- "*10)
#print(item_similarity.shape)

#Predicting User Ratings

item_predicted_ratings = np.dot(property_features.T, item_similarity)
item_predicted_ratings

#Filtering the ratings only for the properties not already rated by the user for recommendation

# np.multiply for cell-by-cell multiplication 

item_final_ratings = np.multiply(item_predicted_ratings, dummy_train)
#item_final_ratings.head()

#EXAMPLE OUTPUT

# Function to get property info based on property_id
def get_property_info(property_id):
    return property_info.loc[property_id]

# Modify the recommendation output code
top_properties = item_final_ratings.iloc[1].sort_values(ascending=False)[0:5]

# Display property name along with propertyId
for property_id, rating in top_properties.items():
    property_name = get_property_info(property_id)['prop_name']
    print(f"Property ID: {property_id}, Property Name: {property_name}, Rating: {rating}")
    print("\n")
    
    
#EVALUATION

test_item_features = X_test.pivot(index = 'propertyId', columns = 'userId', values = 'rating').fillna(0)
test_item_similarity = cosine_similarity(test_item_features)
test_item_similarity[np.isnan(test_item_similarity)] = 0 

#print(test_item_similarity)
#print("- "*10)
#print(test_item_similarity.shape)

item_predicted_ratings_test = np.dot(test_item_features.T, test_item_similarity )
item_predicted_ratings_test

test_item_final_rating = np.multiply(item_predicted_ratings_test, dummy_test)
#test_item_final_rating.head()

ratings['rating'].describe()

#need to normalize the final rating values between range (0.5, 5)

from sklearn.preprocessing import MinMaxScaler

X = test_item_final_rating.copy() 
X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

scaler = MinMaxScaler(feature_range = (0.5, 5))
scaler.fit(X)
pred = scaler.transform(X)

#print(pred)

# total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(pred))
total_non_nan

test = X_test.pivot(index = 'userId', columns = 'propertyId', values = 'rating')
#test.head()

# RMSE Score

diff_sqr_matrix = (test - pred)**2
sum_of_squares_err = diff_sqr_matrix.sum().sum() # df.sum().sum() by default ignores null values

rmse = np.sqrt(sum_of_squares_err/total_non_nan)
print(rmse)

# Mean abslute error

mae = np.abs(pred - test).sum().sum()/total_non_nan
print(mae)