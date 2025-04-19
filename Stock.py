
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, lit
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StockMarketAnalysis") \
    .getOrCreate()

# Load the synthetic data
df = spark.read.csv("synthetic_stock_data_2000.csv", header=True, inferSchema=True)

# Show the first few rows
df.show(5)

# Define a window specification, ordered by Date (since stock data is time series)
window_spec = Window.orderBy("Date").rowsBetween(-4, 0)  # 4 previous rows + the current row for SMA

# Calculate the Simple Moving Average (SMA) with a window size of 5
df = df.withColumn("SMA", avg("Close").over(window_spec))

# Drop rows with null values created by shifting
df = df.dropna()

# Prepare data for regression model
assembler = VectorAssembler(inputCols=["Open", "High", "Low", "Volume", "SMA"], outputCol="features")
df = assembler.transform(df)

# Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Create and train the regression model
lr = LinearRegression(featuresCol="features", labelCol="Close")
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using RMSE (Root Mean Squared Error)
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Evaluate R-squared (R2) to get an idea of how well the model fits
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print(f"R-Squared (R2): {r2}")



# Take user input for prediction
def get_user_input():
    open_price = float(input("Enter the Open price: "))
    high_price = float(input("Enter the High price: "))
    low_price = float(input("Enter the Low price: "))
    volume = int(input("Enter the Volume: "))
    
    # Create a DataFrame with the user input
    user_data = spark.createDataFrame([(open_price, high_price, low_price, volume)], 
                                      ["Open", "High", "Low", "Volume"])
    
    # Calculate SMA for the user input (using the last 5 days from the training data)
    sma = df.select(avg("Close").over(window_spec).alias("SMA")).orderBy("Date", ascending=False).first()[0]
    
    # Add the SMA to the user input
    user_data = user_data.withColumn("SMA", lit(sma))
    
    # Prepare features for prediction
    assembler = VectorAssembler(inputCols=["Open", "High", "Low", "Volume", "SMA"], outputCol="features")
    user_data = assembler.transform(user_data)
    
    # Make prediction using the trained model
    prediction = lr_model.transform(user_data)
    
    # Show the predicted close price
    predicted_close = prediction.select("prediction").first()[0]
    print(f"Predicted Close Price: {predicted_close}")

# Run the user input function
get_user_input()

# Stop the Spark session
spark.stop()
