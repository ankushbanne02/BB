from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Initialize Spark and Load Dataset
spark = SparkSession.builder.appName("SentimentAnalysisLR").getOrCreate()
data_path = "/content/twitter_sentiment_dataset_2000.csv"  # Replace with your dataset path
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Step 2: Preprocess Data (Drop Nulls)
df = df.dropna(subset=["text"])

# Step 3: Convert Sentiment Column to Numeric
df = df.withColumn(
    "sentiment",
    when(col("sentiment") == "Negative", 0)
    .when(col("sentiment") == "Neutral", 1)
    .when(col("sentiment") == "Positive", 2)
    .otherwise(None)
)

# Step 4: Preprocess Data (Tokenization, Stopword Removal, Vectorization)
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")

# Step 5: Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="sentiment", maxIter=10)

# Step 6: Build the Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, lr])

# Step 7: Train-Test Split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 8: Train the Model
model = pipeline.fit(train_data)

# Step 9: Evaluate the Model
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="sentiment", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

# Step 10: Take Input from the User and Make Prediction
def predict_sentiment(user_input):
    # Create a DataFrame from the user input
    input_data = spark.createDataFrame([(user_input,)], ["text"])
    
    # Use the pipeline model to process the input data
    prediction = model.transform(input_data)
    
    # Get the sentiment prediction
    predicted_label = prediction.select("prediction").head()[0]
    sentiment = ["Negative", "Neutral", "Positive"]
    
    # Output the sentiment
    print(f"Predicted Sentiment: {sentiment[int(predicted_label)]}")

# Take input from the user
user_input = input("Enter a tweet to analyze sentiment: ")
predict_sentiment(user_input)

