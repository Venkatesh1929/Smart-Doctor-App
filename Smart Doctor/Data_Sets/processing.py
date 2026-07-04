from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

# ✅ 1. Start Spark Session
spark = SparkSession.builder \
    .appName("SmartDoctorPreprocessing") \
    .config("spark.master", "local[*]") \
    .getOrCreate()

# ✅ 2. Load Dataset
file_path = "C:/Users/vishn/OneDrive/Smart Doctor/Data_Sets/Disease and symptoms dataset.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

print("✅ Dataset loaded successfully")
df.printSchema()
print(f"Total Rows: {df.count()}")

# ✅ 3. Handle Missing Values
df = df.na.fill("Unknown")  # fill missing categorical data
for col_name in df.columns:
    df = df.na.fill({col_name: 0})  # fill numeric nulls with 0

# ✅ 4. Encode Categorical Columns
# Automatically detect string columns
string_cols = [col_name for (col_name, dtype) in df.dtypes if dtype == "string" and col_name != "Disease"]

for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
    df = indexer.fit(df).transform(df)
    df = df.drop(col_name)

# ✅ 5. Label Encode Disease column
label_indexer = StringIndexer(inputCol="Disease", outputCol="label")
df = label_indexer.fit(df).transform(df)

# ✅ 6. Feature Vectorization
feature_cols = [c for c in df.columns if c != "label" and not c.endswith("Disease")]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# ✅ 7. Scale Features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
df = scaler.fit(df).transform(df)

# ✅ 8. Split Train-Test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print("✅ Preprocessing Complete")
print("Training Data:", train_data.count(), "rows")
print("Testing Data:", test_data.count(), "rows")

# ✅ 9. Save processed data to Hadoop or local
train_data.write.mode("overwrite").parquet("hdfs://localhost:9000/smartdoctor/train_data")
test_data.write.mode("overwrite").parquet("hdfs://localhost:9000/smartdoctor/test_data")

print("✅ Data saved to HDFS successfully!")

spark.stop()

