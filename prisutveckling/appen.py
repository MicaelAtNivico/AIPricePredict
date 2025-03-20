from flask import Flask, request, render_template
import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

Country = 'Sweden'

#reading the settings.json file
settings = {}
settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
try:
    with open(settings_path, 'r') as f:
        settings = json.load(f)
except FileNotFoundError:
    print(f"Error: settings.json not found at {settings_path}")
    exit(1)

# Search function to fetch product data
def fetchProduct(search_query):
    url = f'{settings[Country]['fetchProductUrl']}{search_query}'
    #url = f'https://www.pricerunner.se/se/api/search-compare-gateway/public/search/suggest/SE?q={search_query}'
    response = requests.get(url).content
    data = json.loads(response)
    return data

# Function to fetch product price history
def fetchPriceHistory(product_id):
    url = f'{settings[Country]['fetchHistoryUrlP1']}{product_id}{settings[Country]['fetchHistoryUrlP2']}'
    #url = f'https://www.pricerunner.se/se/api/search-compare-gateway/public/pricehistory/product/{product_id}/SE/DAY?merchantId=&selectedInterval=INFINITE_DAYS&filter=NATIONAL'

    response = requests.get(url).content
    data = json.loads(response)
    price_history = pd.DataFrame(data['history'])
    return price_history

# AI prediction function
def aiPredict(price_history):
    price_history['timestamp'] = pd.to_datetime(price_history['timestamp'], utc=True)
    price_history = price_history.sort_values(by='timestamp')

    # Create a target column: 1 if price increased, 0 if decreased
    price_history['price_change'] = (price_history['price'].diff() > 0).astype(int)
    price_history = price_history.dropna()

    # Feature engineering: Add lag features
    price_history['price_lag1'] = price_history['price'].shift(1)
    price_history['price_lag2'] = price_history['price'].shift(2)
    price_history = price_history.dropna()

    # Select features and target
    features = ['price_lag1', 'price_lag2']
    target = 'price_change'

    X = price_history[features]
    y = price_history[target]

    # Train a Random Forest Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict future prices
    future_prices = []
    current_features = price_history.iloc[-1][features].values
    for _ in range(30):
        current_features_df = pd.DataFrame([current_features], columns=features)
        next_price_change = model.predict(current_features_df)[0]
        next_price = current_features[0] + (1 if next_price_change == 1 else -1) * 10
        future_prices.append(next_price)
        current_features = np.roll(current_features, -1)
        current_features[-1] = next_price

    # Generate dates for the next 30 days
    future_dates = pd.date_range(price_history['timestamp'].iloc[-1] + pd.Timedelta(days=1), periods=30)

    # Prepare data for the frontend
    historical_data = {
        'dates': price_history['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        'prices': price_history['price'].tolist()
    }
    predicted_data = {
        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
        'prices': future_prices
    }

    return historical_data, predicted_data

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Search page
@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search']
    result = fetchProduct(search_query)
    products = result.get('products', [])
    return render_template('search.html', products=products)

# Check page
@app.route('/check')
def check():
    product_id = request.args.get('id')
    price_history = fetchPriceHistory(product_id)
    historical_data, predicted_data = aiPredict(price_history)
    return render_template('check.html', historical_data=historical_data, predicted_data=predicted_data)

if __name__ == '__main__':
    app.run(debug=True)