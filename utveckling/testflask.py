from flask import Flask, request, render_template_string
import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Search function to fetch product data
def fetchProduct(search_query):
    url = f'https://www.pricerunner.se/se/api/search-compare-gateway/public/search/suggest/SE?q={search_query}'
    response = requests.get(url).content
    data = json.loads(response)
    return data

# Function to fetch product price history
def fetchPriceHistory(product_id):
    url = f'https://www.pricerunner.se/se/api/search-compare-gateway/public/pricehistory/product/{product_id}/SE/DAY?merchantId=&selectedInterval=INFINITE_DAYS&filter=NATIONAL'
    response = requests.get(url).content
    data = json.loads(response)
    price_history = pd.DataFrame(data['history'])
    return(price_history)

def aiPredict(price_history):
    price_history['timestamp'] = pd.to_datetime(price_history['timestamp'], utc=True)

    price_history = price_history.sort_values(by='timestamp')

    # Create a target column: 1 if price increased, 0 if decreased
    price_history['price_change'] = (price_history['price'].diff() > 0).astype(int)

    # Drop the first row with NaN in price_change
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

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    future_prices = []
    current_features = price_history.iloc[-1][features].values  # Start with the latest features

    for _ in range(30):  # Predict for the next 30 days
        # Convert current_features to a DataFrame with the same feature names
        current_features_df = pd.DataFrame([current_features], columns=features)
        
        # Predict the next price change
        next_price_change = model.predict(current_features_df)[0]
        
        # Adjust price (example: ±10)
        next_price = current_features[0] + (1 if next_price_change == 1 else -1) * 10
        future_prices.append(next_price)
        
        # Update features for the next prediction
        current_features = np.roll(current_features, -1)  # Shift features
        current_features[-1] = next_price  # Add the predicted price as the latest feature

    # Generate dates for the next 30 days
    future_dates = pd.date_range(price_history['timestamp'].iloc[-1] + pd.Timedelta(days=1), periods=30)

    # Plot the predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(price_history['timestamp'], price_history['price'], label='Historical Prices')
    plt.plot(future_dates, future_prices, label='Predicted Prices', linestyle='--', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Next Month Price Prediction')
    plt.legend()
    # plt.show()
    plt.savefig('static/image.png')

# Main page
@app.route('/')
def index():
    return '''
        <form action="/search" method="post">
            <label for="search">Search:</label>
            <input type="text" id="search" name="search" required>
            <button type="submit">Submit</button>
        </form>
    '''

# Search page
@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search']
    result = fetchProduct(search_query)

    # Extract products from the result
    products = result.get('products', [])

    # Render the results as clickable links
    html = '''
        <h1>Search Results</h1>
        <ul>
    '''
    for product in products:
        product_name = product['name']
        product_id = product['id']
        html += f'<li><a href="/check?id={product_id}">{product_name}</a></li>'
    html += '''
        </ul>
        <a href="/">Go back</a>
    '''
    return html

# Generating price prediction - and displaying the graph
@app.route('/check')
def check():
    product_id = request.args.get('id')
    price_history = fetchPriceHistory(product_id)
    image = aiPredict(price_history)
    html = '''
        <h1>Price Prediction</h1>
        <img src="/static/image.png" alt="Price Prediction Graph">
        <br><br>
        <a href="/">Go back</a>
    '''
    return html


if __name__ == '__main__':
    app.run(debug=True)