import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, LinearSVC, LogisticRegression, MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.sql.functions.{concat_ws, lit}
import org.apache.spark.sql.functions.{lower, regexp_replace}


object hw1 {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "E://" + "Programming/hadoop")

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("pokrovsky")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    val dataSetRaw = spark.read
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("escape", "\"")
      .csv("datasets/train.csv")

    val dataSet = dataSetRaw
      .withColumn("target", dataSetRaw.col("target").cast("Integer"))
      .na.fill("no_location_info", Array("location"))
      .na.fill("no_keyword_info", Array("keyword"))
    dataSet.show()

    var df = dataSet.select(dataSet("target").as("label"), concat_ws(" ", dataSet("keyword"), dataSet("location"), dataSet("text")).as("new_text"))
    df = df.withColumn("new_text", lower(df.col("new_text")))
    df.show()
    val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed = 12345)
    val tokenizer = new Tokenizer()
      .setInputCol("new_text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(350000)



    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    val lr = new LogisticRegression()
          .setElasticNetParam(0.4)
          .setRegParam(0.01)
          .setFamily("binomial")







    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, idf,  lr))
    val model = pipeline.fit(training)

    val resRaw = model.transform(test)
    val res = resRaw.withColumn("prediction", resRaw.col("prediction").cast("Integer"))
    val resToEval = res.select("label", "prediction")
    val accuracyCount = resToEval.where(resToEval("label") ===  resToEval("prediction")).count()
    val resCount = resToEval.count()
    println("test accuracy: "+ accuracyCount.asInstanceOf[Double]/resCount)

    val trainRaw = model.transform(training)
    val trainRes = trainRaw.withColumn("prediction", trainRaw.col("prediction").cast("Integer"))
    val trainResToEval = trainRes.select("label", "prediction")
    val accuracyTrainCount = trainResToEval.where(trainResToEval("label") ===  trainResToEval("prediction")).count()
    val resTrainCount = trainResToEval.count()
    println("Train accuracy: "+ accuracyTrainCount.asInstanceOf[Double]/resTrainCount)

    val truePositive = resToEval.where(resToEval("label") === 1 && resToEval("prediction") === 1).count()
    val falsePositive = resToEval.where(resToEval("label") === 0 && resToEval("prediction") === 1).count()
    val falseNegative = resToEval.where(resToEval("label") === 1 && resToEval("prediction") === 0).count()
    val precision = truePositive.asInstanceOf[Double]/(truePositive+falsePositive)
    val recall = truePositive.asInstanceOf[Double]/(truePositive+falseNegative)
    val f_score = 2*precision*recall/(precision+recall)
    println("f score: "+f_score)
    val submitDfRaw = spark.read
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("escape", "\"")
      .csv("datasets/test.csv")
      .withColumn("label", lit(0))
      .na.fill("no_location_info", Array("location"))
      .na.fill("no_keyword_info", Array("keyword"))

    var submitDf = submitDfRaw.select(submitDfRaw("id").as("raw_id"), submitDfRaw("label"), concat_ws(" ", submitDfRaw("keyword"), submitDfRaw("location"), submitDfRaw("text")).as("new_text"))
    submitDf= submitDf.withColumn("new_text", lower(submitDf.col("new_text")))
    val submit = model.transform(submitDf)
    var submission = spark.read
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("escape", "\"")
      .csv("datasets/sample_submission.csv")
    submission = submission.join(submit, submission("id")===submit("raw_id"), "left")

      submission = submission
        .drop("target")
        .withColumn("target", submission.col("prediction").cast("Integer"))


    submission = submission.select(submission("id"), submission("target"))
    submission
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .mode("overwrite")
      .option("delimiter", "\t")
      .option("sep", ";")
      .save("mydata.csv")

  }

}
