from pyspark.sql import SparkSession

sc=SparkSession.builder.appName("Word_Count").master("local").getOrCreate()

text_rdd=sc.sparkContext.textFile("ASD.txt")

word_rdd=text_rdd.flatMap(lambda line:line.split(" "))

print(word_rdd.collect())

word_flat=word_rdd.map(lambda word:(word,1))
print(word_flat.collect())

word_count=word_flat.reduceByKey(lambda x,y:x+y)
print(word_count.collect())

sc.stop()


from pyspark import SparkContext

sc=SparkContext("local","word_count");

word_rdd= sc.textFile("ASD.txt")



word_flat=word_rdd.flatMap(lambda line:line.split(" "))

word_map=word_flat.map(lambda word:(word,1))

word_count=word_map.reduceByKey(lambda x,y:x+y)

print(word_count.collect())

sc.stop()