from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Load data and preprocess
ratings = pd.read_csv('ratings2.csv')
property_info = pd.read_csv('propertiesprocessed.csv')
property_info.set_index('property_id', inplace=True)
users = pd.read_csv('userinfo.csv')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username'].strip()
    password = request.form['password'].strip()

    print(f"Entered Username: {username}, Password: {password}")

    # Convert the 'password' column to strings to use .str accessor
    users['password'] = users['password'].astype(str)

    # Check if the entered credentials exist in userinfo.csv after stripping whitespaces
    user_info = users[(users['username'].str.strip() == username) & (users['password'].str.strip() == password)]
    print("User Info DataFrame:")
    print(user_info)

    if not user_info.empty:
        user_id = int(user_info.iloc[0]['user_id'])  # Convert int64 to int
        session['user_id'] = user_id
        print(f"Successful login. Redirecting to index.html for User ID: {user_id}")
        return render_template('index.html')
    else:
        print("Invalid credentials!")
        return render_template('login.html', error='Invalid credentials')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))
    
@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    user_recommendations = get_recommendations()
    
    return render_template('index.html', user_id=session['user_id'], recommendations=user_recommendations)

@app.route('/get_recommendations', methods=['GET', 'POST'])
def get_recommendations():
    if request.method == 'POST':
        user_id = session['user_id']
        
        # Load data and preprocess
        ratings = pd.read_csv('ratings2.csv')
        property_info = pd.read_csv('propertiesprocessed.csv')
        property_info.set_index('property_id', inplace=True)
        
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

    # Save the updated ratings to ratings2.csv
    ratings.to_csv('ratings2.csv', index=False)

    return render_template('rate.html')

@app.route('/view_ratings', methods=['GET', 'POST'])
def view_ratings():
    if request.method == 'POST':
        user_id = session['user_id']
    
        # Get all rated properties for the user
        user_ratings = ratings[(ratings['userId'] == user_id) & (ratings['rating'] > 0)]

        # Merge with property_info to get property details
        rated_properties = pd.merge(user_ratings, property_info, how='inner', left_on='propertyId', right_index=True)

        # Sort by rating in descending order
        rated_properties = rated_properties.sort_values(by='rating', ascending=False)

        return render_template('view_ratings.html', user_id=user_id, rated_properties=rated_properties)

    return render_template('view_ratings_input.html')

@app.route('/view_residences', methods=['GET', 'POST'])
def search_property_name():
    if request.method == 'POST':
        user_id = session['user_id']
        user_input = request.form['user_input']
        region_filter = request.form['region_filter']
        location_filter = request.form['location_filter']
        threshold = 80
        
        # Load data from 'propertiesprocessed.csv'
        data = pd.read_csv('propertiesprocessed.csv')
        
        # If user_input is empty, create a list of all property names
        if not user_input:
            property_names = data['prop_name'].tolist()
        else:
            # Use fuzzywuzzy process.extract to get matches with a score
            matches = process.extract(user_input, data['prop_name'])
            
            # Filter matches based on the threshold score
            filtered_matches = [match for match in matches if match[1] >= threshold]

            # Extract property names from filtered matches
            property_names = [data.loc[match[2], 'prop_name'] for match in filtered_matches]

            # Extract property names, scores, regions, and locations from filtered matches
            results = [{'property_id': data.loc[data['prop_name'] == match[0], 'property_id'].iloc[0],
                        'prop_name': match[0],
                        'score': match[1],
                        'region': data.loc[data['prop_name'] == match[0], 'region'].iloc[0],
                        'location': data.loc[data['prop_name'] == match[0], 'location'].iloc[0],
                        'monthly_rent': data.loc[data['prop_name'] == match[0], 'monthly_rent'].iloc[0],
                        'property_type': data.loc[data['prop_name'] == match[0], 'property_type'].iloc[0],
                        'rooms': data.loc[data['prop_name'] == match[0], 'rooms'].iloc[0],
                        'parking': data.loc[data['prop_name'] == match[0], 'parking'].iloc[0],
                        'bathroom': data.loc[data['prop_name'] == match[0], 'bathroom'].iloc[0],
                        'size': data.loc[data['prop_name'] == match[0], 'size'].iloc[0],
                        'furnished': data.loc[data['prop_name'] == match[0], 'furnished'].iloc[0],
                        'facilities_provided': data.loc[data['prop_name'] == match[0], 'facilities_provided'].iloc[0]}
                    for match in filtered_matches]

            # Filter results based on region if specified by the user
            if region_filter and len(region_filter) > 0:
                results = [result for result in results if result['region'] == region_filter]

            # Filter results based on location if specified by the user
            if location_filter and len(location_filter) > 0:
                results = [result for result in results if result['location'] == location_filter]

            return render_template('view_property.html', user_id=user_id, results=results)

    return render_template('search_property.html')


if __name__ == '__main__':
    app.run(debug=True)
