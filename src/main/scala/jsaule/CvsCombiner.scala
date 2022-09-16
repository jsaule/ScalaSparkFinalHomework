package jsaule

import jsaule.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.functions.{col, lit}

/**
 * object for combining 2 CSV files
 * with only required columns for analysis:
 * date, open, high, low, close, volume, ticker
 * */
object CvsCombiner extends App {

  val spark = getSpark("Sparky")

  // getting the ticker from filepath

  val filePath1 = "src/resources/NVDA.csv"
  val ticker1 = filePath1.split("/").last.take(4)

  val filePath2 = "src/resources/TWTR.csv"
  val ticker2 = filePath2.split("/").last.take(4)

  // saving dataframes

  val dfOriginal1 = readDataWithView(spark, filePath1)
  val dfOriginal2 = readDataWithView(spark, filePath2)

  dfOriginal1.cache()
  dfOriginal2.cache()

  // dropping column 'Adj close' and adding column 'ticker'

  val dfTicker1 = dfOriginal1.drop("Adj close").withColumn("ticker", lit(ticker1))
  val dfTicker2 = dfOriginal2.drop("Adj close").withColumn("ticker", lit(ticker2))

  // changing all column names to lowercase

  val df1 = dfTicker1.select(dfTicker1.columns.map(x => col(x).as(x.toLowerCase)): _*)
  val df2 = dfTicker2.select(dfTicker2.columns.map(x => col(x).as(x.toLowerCase)): _*)

  // combining the two dataframes into one

  val df = df1.union(df2).orderBy("date", "ticker")
  df.sample(0.5).show(40)

  val outputPath = "src/resources/twitter_nvidia.csv"
  df.write.mode("overwrite").option("header", true).csv(outputPath)

}
