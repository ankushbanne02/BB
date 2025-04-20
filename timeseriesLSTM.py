import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, unix_timestamp, hour, dayofweek, col

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("LSTM_TrafficPrediction") \
    .getOrCreate()

# 2. Load and preprocess data
df = spark.read.csv("traffic_data_extended.csv", header=True, inferSchema=True)
df = df.withColumn("timestamp", to_timestamp("timestamp"))
df = df.filter(df["timestamp"].isNotNull() & df["traffic_volume"].isNotNull())

# 3. Feature engineering
df = df.withColumn("hour", hour("timestamp")) \
       .withColumn("day_of_week", dayofweek("timestamp")) \
       .withColumn("timestamp_unix", unix_timestamp("timestamp").cast("double")) \
       .withColumn("is_weekend", ((col("day_of_week") == 1) | (col("day_of_week") == 7)).cast("int"))

# 4. Convert to pandas for LSTM
pandas_df = df.toPandas()

# Sort by timestamp to ensure proper sequence for time series prediction
pandas_df = pandas_df.sort_values('timestamp')

# 5. Normalize features and traffic volume for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
pandas_df[['hour', 'day_of_week', 'temperature', 'rainfall', 'is_weekend']] = scaler.fit_transform(
    pandas_df[['hour', 'day_of_week', 'temperature', 'rainfall', 'is_weekend']])

traffic_volume_scaler = MinMaxScaler(feature_range=(0, 1))
pandas_df['traffic_volume'] = traffic_volume_scaler.fit_transform(pandas_df[['traffic_volume']])

# 6. Prepare the data for LSTM: Convert time series data into sequences
def create_sequence_data(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # Features (all columns except the target)
        y.append(data[i+seq_length, -1])  # Target (traffic_volume)
    return np.array(X), np.array(y)

# Select the columns for features and target
features = pandas_df[['hour', 'day_of_week', 'temperature', 'rainfall', 'is_weekend']].values
target = pandas_df['traffic_volume'].values

# Set sequence length (number of previous time steps to consider for prediction)
seq_length = 24  # 24 hours as input for predicting the next traffic volume

# Prepare data for LSTM
X, y = create_sequence_data(np.hstack((features, target.reshape(-1, 1))), seq_length)

# 7. Reshape data for LSTM input
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # [samples, time steps, features]

# 8. Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 9. Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)

# 10. Make predictions
predictions = model.predict(X)

# Inverse scale the predictions and actual values
predictions = traffic_volume_scaler.inverse_transform(predictions)
y_actual = traffic_volume_scaler.inverse_transform(y.reshape(-1, 1))

# 11. Plot the results
plt.figure(figsize=(14, 6))
plt.plot(pandas_df['timestamp'][seq_length:], y_actual, label='Actual Traffic Volume', alpha=0.5)
plt.plot(pandas_df['timestamp'][seq_length:], predictions, label='Predicted Volume', color='red', alpha=0.8)
plt.title("Traffic Volume Prediction Using LSTM")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# 12. Make a sample future prediction with user input (assuming LSTM works well)
# Collect user input for future prediction
hour_input = int(input("Hour of Day (0–23): "))
day_input = int(input("Day of Week (1=Sun, 7=Sat): "))
temp_input = float(input("Temperature (°C): "))
rain_input = float(input("Rainfall (mm): "))
weekend_input = int(input("Is Weekend? (0=No, 1=Yes): "))

# Scale the input values using the same scaler used for training
user_input_scaled = scaler.transform([[hour_input, day_input, temp_input, rain_input, weekend_input]])

# Reshape input for LSTM (same length of sequence as training data)
user_input_scaled = np.repeat(user_input_scaled, seq_length, axis=0).reshape(1, seq_length, -1)

# Predict future traffic volume
future_prediction = model.predict(user_input_scaled)
predicted_value = traffic_volume_scaler.inverse_transform(future_prediction)

print(f"\n🚗 Predicted Traffic Volume: {predicted_value[0][0]:.2f} vehicles")

# 13. Stop Spark session
spark.stop()
