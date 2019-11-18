from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.corpus import stopwords
import nltk
import pandas
from sklearn.externals import joblib
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics



def train(model, model_name):
    pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx,model])
    (trainingData, testData) = df.randomSplit([0.7, 0.3], seed = 100)
    pipelineFit = pipeline.fit(trainingData)
    predictions = pipelineFit.transform(testData)
    evaluator = MulticlassClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    predictions.show(20)
    path= model_name+".model"
    pipelineFit.write().overwrite().save(path)
    print(model_name+": ",str(accuracy))
    predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda r : (r[0], r[1]))
    metrics = MulticlassMetrics(predictionAndLabels)

    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()

    print("Summary:")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

if __name__ == "__main__":
    sc =SparkContext()
    sqlContext = SQLContext(sc)
    sc.setLogLevel("WARN")

    data = sc.textFile('texts.txt')
    records = data.map(lambda row: row.split('||'))
    filtered_records = records.filter(lambda line: len(line) == 2)
    header = 'category text'
    df = filtered_records.toDF()
    regexTokenizer = RegexTokenizer(inputCol="_2", outputCol="words", pattern="\\W")
    add_stopwords = stopwords.words('english')
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) 
    label_stringIdx = StringIndexer(inputCol = "_1", outputCol = "label").setHandleInvalid("keep")
    print("Model: Logistic Regression")
    lr = LogisticRegression(maxIter=50, regParam=0.3, elasticNetParam=0)
    train(lr, "lr")
    print("Model: Random Forest")
    rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)
    train(rf, "rf")
    print("Model: Naive Bayes")
    nb = NaiveBayes(smoothing=1)
    train(nb, "nb")

