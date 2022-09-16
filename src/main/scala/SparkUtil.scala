import org.apache.spark.sql.{DataFrame, SparkSession}

object SparkUtil extends App {
  /**
   * Returns a new or an existing Spark session
   *
   * @param appName        - name of our Spark instance
   * @param partitionCount default 5 - starting default is 200
   * @param master         default "local"  - master URL to connect
   * @param verbose        - prints debug info
   * @return sparkSession
   */
  def getSpark(appName: String, partitionCount: Int = 1, master: String = "local", verbose: Boolean = true): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")
    val sparkSession = SparkSession.builder().appName(appName).master(master).getOrCreate()
    sparkSession.conf.set("spark.sql.shuffle.partitions", partitionCount)
    if (verbose) println(s"Session started on Spark version ${sparkSession.version} with ${partitionCount} partition(s)")
    sparkSession
  }

  //TODO write scalaDoc
  def readDataWithView(spark: SparkSession,
                       filePath: String,
                       source: String = "csv",
                       viewName: String = "dfTable",
                       header: Boolean = true,
                       inferSchema: Boolean = true,
                       printSchema: Boolean = true,
                       cacheOn: Boolean = true
                      ): DataFrame = {

    val df = spark.read.format(source)
      .option("header", header.toString) //Spark wants string here since option is generic
      .option("inferSchema", inferSchema.toString) //we let Spark determine schema
      .load(filePath)
    //so if you pass only whitespace or nothing to view we will not create it
    //so if viewName is NOT blank
    if (viewName.nonEmpty) {
      df.createOrReplaceTempView(viewName)
      println(s"Created Temporary View for SQL queries called: $viewName")
    }
    if (printSchema) df.printSchema()
    if (cacheOn) df.cache()
    df
  }
}