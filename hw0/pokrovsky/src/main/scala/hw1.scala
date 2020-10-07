import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._



object hw1 {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "E://" + "Programming/hadoop")

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("pokrovsky")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    val dataSet = spark.read
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("escape", "\"")
      .csv("datasets/")


    val features = dataSet
      .withColumn("price", dataSet("price").cast("Integer"))
      .withColumn("latitude", dataSet("latitude").cast("Double"))
      .withColumn("longitude", dataSet("longitude").cast("Double"))
      .withColumn("minimum_nights", dataSet("minimum_nights").cast("Integer"))
      .withColumn("number_of_reviews", dataSet("number_of_reviews").cast("Integer"))
      .where(col("price") > 0)
      .na
      .drop()

    features.createOrReplaceTempView("features")  // Able to work with sql


    println("Mean")
    features
      .groupBy("room_type")
      .mean("price")
      .show()

    println("Median")
    spark.sql("select room_type, percentile_approx(price, 0.5) as median from features group by room_type ").show()

    println("Mode")
    val room_features = features
      .groupBy("room_type", "price")
      .count()
    val windowSpec = Window.partitionBy("room_type").orderBy(desc("count"))
    room_features.withColumn("row_number", row_number().over(windowSpec))
      .select("room_type", "price")
      .where(col("row_number") === 1)
      .show()



    println("Standard deviation")
    features.select("room_type", "price")
      .groupBy("room_type")
      .agg(stddev("price"))
      .show()

    println("max price")
    features.orderBy(desc("price")).show(1)

    println("min price")
    features.orderBy("price").show(1)

    println("Correlation between price and minimum_nights")
    println(features.stat.corr("price", "minimum_nights", "pearson"))
    println("Correlation between price and number_of_reviews")
    println(features.stat.corr("price", "number_of_reviews", "pearson"))

    // Стырено https://github.com/mumoshu/geohash-scala
    val encode = ( lat:Double, lng:Double, precision:Int )=> {
      val base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
      var (minLat,maxLat) = (-90.0,90.0)
      var (minLng,maxLng) = (-180.0,180.0)
      val bits = List(16,8,4,2,1)

      (0 until precision).map{ p => {
        base32 apply (0 until 5).map{ i => {
          if (((5 * p) + i) % 2 == 0) {
            val mid = (minLng+maxLng)/2.0
            if (lng > mid) {
              minLng = mid
              bits(i)
            } else {
              maxLng = mid
              0
            }
          } else {
            val mid = (minLat+maxLat)/2.0
            if (lat > mid) {
              minLat = mid
              bits(i)
            } else {
              maxLat = mid
              0
            }
          }
        }}.reduceLeft( (a,b) => a|b )
      }}.mkString("")
    }

    val encode_udf = udf(encode)
    val features_geohash = features
        .withColumn("geoHash", encode_udf(col("latitude"), col("longitude"), lit(5)))
        .groupBy("geoHash")
        .mean("price")
        .orderBy(desc("avg(price)"))
        .first()


    val point = GeoHash.decode(features_geohash.get(0).toString)
    println("The most expensive area 5x5 km in New-York in square with center in "+point+" with mean price: "+features_geohash.get(1))



  }




}
