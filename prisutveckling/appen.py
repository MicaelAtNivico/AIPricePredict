from flask import Flask, request, render_template, redirect, url_for
import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

Country = 'Sweden'

# Reading the settings.json file
settings = {}
settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
try:
    with open(settings_path, 'r') as f:
        settings = json.load(f)
except FileNotFoundError:
    print(f"Error: settings.json not found at {settings_path}")
    exit(1)

# Initialize SQLite database
db_path = os.path.join(os.path.dirname(__file__), 'history.db')
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_query TEXT,
    product_id TEXT,
    fetched_data TEXT,
    predicted_data TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# Search function to fetch product data
def fetchProduct(search_query):
    url = f'{settings[Country]["fetchProductUrl"]}{search_query}'
    response = requests.get(url).content
    data = json.loads(response)
    return data

# Function to fetch product price history
def fetchPriceHistory(product_id):
    url = f'{settings[Country]["fetchHistoryUrlP1"]}{product_id}{settings[Country]["fetchHistoryUrlP2"]}'
    response = requests.get(url).content
    data = json.loads(response)
    price_history = pd.DataFrame(data['history'])

    # Detect and handle outliers
    Q1 = price_history['price'].quantile(0.25)
    Q3 = price_history['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with the median price
    median_price = price_history['price'].median()
    price_history['price'] = price_history['price'].apply(
        lambda x: median_price if x < lower_bound or x > upper_bound else x
    )

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

    # Calculate AI accuracy
    accuracy = model.score(X_test, y_test)

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

    # Creating data for the frontend
    historical_data = {
        'dates': price_history['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        'prices': price_history['price'].tolist()
    }
    predicted_data = {
        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
        'prices': future_prices
    }

    return historical_data, predicted_data, accuracy

# Main page
@app.route('/')
def index():
    cursor.execute('SELECT id, search_query, timestamp FROM searches ORDER BY timestamp DESC')
    history_entries = cursor.fetchall()
    return render_template('index.html', history_entries=history_entries)

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
    search_query = request.args.get('query') 
    price_history = fetchPriceHistory(product_id)
    historical_data, predicted_data, accuracy = aiPredict(price_history)

    # Save to database
    cursor.execute('''
        INSERT INTO searches (search_query, product_id, fetched_data, predicted_data)
        VALUES (?, ?, ?, ?)
    ''', (search_query, product_id, json.dumps(historical_data), json.dumps(predicted_data)))
    conn.commit()

    return render_template('check.html', historical_data=historical_data, predicted_data=predicted_data, accuracy=accuracy)

# Detailed history for old searches
@app.route('/history/<int:history_id>')
def history_detail(history_id):
    # Fetch the saved data from the database
    cursor.execute('SELECT search_query, product_id, fetched_data, predicted_data FROM searches WHERE id = ?', (history_id,))
    entry = cursor.fetchone()
    if entry:
        search_query, product_id, fetched_data, predicted_data = entry

        # Fetch the latest price history from PriceRunner
        latest_price_history = fetchPriceHistory(product_id)

        # UTC
        latest_price_history['timestamp'] = pd.to_datetime(latest_price_history['timestamp'], errors='coerce', utc=True)

        # saved predicted data
        predicted_data = json.loads(predicted_data)

        # latest historical data for the chart
        latest_historical_data = {
            'dates': latest_price_history['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': latest_price_history['price'].tolist()
        }

        return render_template(
            'history_detail.html',
            search_query=search_query,
            latest_historical_data=latest_historical_data,
            predicted_data=predicted_data
        )
    else:
        return "History entry not found", 404

if __name__ == '__main__':
    app.run(debug=True)