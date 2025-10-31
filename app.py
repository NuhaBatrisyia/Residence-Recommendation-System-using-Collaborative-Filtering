from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data and preprocess
ratings = pd.read_csv('ratings.csv')
property_info = pd.read_csv('propertiesprocessed.csv')
property_info.set_index('property_id', inplace=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['GET', 'POST'])
def get_recommendations():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])

        X_train, X_test = train_test_split(ratings, test_size=0.30, random_state=42)

        # pivot ratings into movie features
        user_data = X_train.pivot(index='userId', columns='propertyId', values='rating').fillna(0)
        dummy_train = X_train.copy()
        dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
        dummy_train = dummy_train.pivot(index='userId', columns='propertyId', values='rating').fillna(1)

        property_features = X_train.pivot(index='propertyId', columns='userId', values='rating').fillna(0)
        item_similarity = cosine_similarity(property_features)
        item_similarity[np.isnan(item_similarity)] = 0

        item_predicted_ratings = np.dot(property_features.T, item_similarity)
        item_final_ratings = np.multiply(item_predicted_ratings, dummy_train)

        unrated_properties = user_data.loc[user_id][user_data.loc[user_id] == 0].index
        top_properties = item_final_ratings.iloc[user_id].sort_values(ascending=False)[0:5]
        top_unrated_properties = top_properties[top_properties.index.isin(unrated_properties)].head(5)

        recommendations = []
        for property_id, rating in top_unrated_properties.items():
            property_name = property_info.loc[property_id, 'prop_name']
            monthly_rent = property_info.loc[property_id, 'monthly_rent']
            location = property_info.loc[property_id,'location']
            region = property_info.loc[property_id,'region']
            property_type = property_info.loc[property_id,'property_type']
            rooms = property_info.loc[property_id,'rooms']
            bathroom = property_info.loc[property_id,'bathroom']
            parking = property_info.loc[property_id,'parking']
            size = property_info.loc[property_id,'size']
            furnished = property_info.loc[property_id,'furnished']
            facilities_provided = property_info.loc[property_id,'facilities_provided']

            recommendations.append({
                'property_id': property_id,
                'property_name': property_name,
                'rating': rating,
                'monthly_rent': monthly_rent,
                'location': location,
                'property_type': property_type,
                'region': region,
                'rooms': rooms,
                'bathroom': bathroom,
                'parking': parking,
                'size': size,
                'furnished': furnished,
                'facilities_provided': facilities_provided
            })

        return render_template('recommendations.html', user_id=user_id, recommendations=recommendations)

    return render_template('get_recommendations_input.html')

@app.route('/rate', methods=['POST'])
def rate():
    global ratings  # Declare ratings as a global variable

    user_id = int(request.form['user_id'])
    property_id = int(request.form['property_id'])
    rating = int(request.form['rating'])
    
    # Update ratings DataFrame
    new_rating = pd.DataFrame({'userId': [user_id], 'propertyId': [property_id], 'rating': [rating]})
    ratings = pd.concat([ratings, new_rating], ignore_index=True)

    # Save the updated ratings to ratings.csv
    ratings.to_csv('ratings.csv', index=False)

    return render_template('rate.html')

@app.route('/view_ratings', methods=['GET', 'POST'])
def view_ratings():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])

        # Get all rated properties for the user
        user_ratings = ratings[(ratings['userId'] == user_id) & (ratings['rating'] > 0)]

        # Merge with property_info to get property details
        rated_properties = pd.merge(user_ratings, property_info, how='inner', left_on='propertyId', right_index=True)

        # Sort by rating in descending order
        rated_properties = rated_properties.sort_values(by='rating', ascending=False)

        return render_template('view_ratings.html', user_id=user_id, rated_properties=rated_properties)

    return render_template('view_ratings_input.html')

if __name__ == '__main__':
    app.run(debug=True)
