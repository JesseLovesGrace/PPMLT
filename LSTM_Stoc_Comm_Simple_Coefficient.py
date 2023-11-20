import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i + sequence_length]
        target = data.iloc[i + sequence_length]['close']
        sequences.append(sequence.values)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Function to build and train LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to find correlation and train LSTM model
def find_correlation_and_train(stock_file, commodity_file):
    # Load and preprocess data
    stock_data = load_and_preprocess_data(stock_file)
    commodity_data = load_and_preprocess_data(commodity_file)

    # Merge data on date
    merged_data = pd.merge(stock_data, commodity_data, how='inner', on='date')

    # Calculate daily returns
    merged_data['stock_return'] = merged_data['close_x'].pct_change()
    merged_data['commodity_return'] = merged_data['close_y'].pct_change()

    # Drop NaN values
    merged_data = merged_data.dropna()

    # Calculate correlation coefficient
    correlation_coefficient, _ = pearsonr(merged_data['stock_return'], merged_data['commodity_return'])
    print(f"Correlation Coefficient: {correlation_coefficient}")

    # Prepare data for LSTM model
    sequence_length = 10  # You can adjust this based on your preference
    X, y = create_sequences(merged_data[['stock_return', 'commodity_return']], sequence_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train LSTM model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate model on test data
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean Squared Error on Test Data: {mse}")

    # Plot predictions vs actual values
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

    # Display the model summary
    model.summary()

# Specify your file paths
stock_folder = r'C:\Users\jesse\Desktop\PPMLT\Stocks'
commodity_folder = r'C:\Users\jesse\Desktop\PPMLT\Commodities'

# Loop through each stock file and pair with each commodity file
for stock_file in os.listdir(stock_folder):
    stock_file_path = os.path.join(stock_folder, stock_file)
    
    for commodity_file in os.listdir(commodity_folder):
        commodity_file_path = os.path.join(commodity_folder, commodity_file)

        # Perform analysis for each pair
        print(f"\nAnalyzing pair: {stock_file} - {commodity_file}")
        find_correlation_and_train(stock_file_path, commodity_file_path)
