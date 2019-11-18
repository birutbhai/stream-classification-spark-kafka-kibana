from kafka import KafkaConsumer
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import Row
import time
from pyspark.mllib.evaluation import MulticlassMetrics
from elasticsearch import Elasticsearch
es=Elasticsearch([{'host':'localhost','port':9200}])
sc =SparkContext()
sqlContext = SQLContext(sc)
consumer = KafkaConsumer('test',
                         group_id='my-group',
                         bootstrap_servers=['localhost:9092'])
evaluator = MulticlassClassificationEvaluator()
count=0;
sum=[0,0,0];
avg=0;
t =1
for message in consumer:
    consumer.commit() 
    message=message.value.decode('utf-8')
    print(message)
    category='"'+message.split("||")[0]
    text=message.split("||")[1]+'"'
    message=sc.parallelize([(category,text)])
    records = message.map(lambda row: Row(_1=row[0],_2=row[1]))
    df=sqlContext.createDataFrame(records)
    models = ["lr.model", "nb.model"]#, "rf.model"]
    count+=1
    j = 0
    for model in models:
        pipeline_model = PipelineModel.load(model)
        predictions = pipeline_model.transform(df)
        predictions.show()
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda r : (r[0], r[1]))
        metrics = MulticlassMetrics(predictionAndLabels)
        precision = metrics.precision()
        recall = metrics.recall()
        f1Score = metrics.fMeasure()
        print("Summary Stats with "+ model)
        print("Accuracy:",accuracy)
        print("Precision = %s" % precision)
        print("Recall = %s" % recall)
        print("F1 Score = %s" % f1Score)
        sum[j]+=accuracy
        #test = predictions.show()
        #print("type:", str(test))
        #print("coulmn:",predictions.collect())
        e1={"Classifier":model, "Accuracy":accuracy, "Precision":precision, "Recall":recall, "F1":f1Score}
        res = es.index(index='news_classification_'+model,doc_type=model,id=t,body=e1)
        print("Res:"+str(res['result'])+" id "+str(t))
        t = t+1
        if(count%10==0):
            e = {"Classifier":model,"Batch_Accuracy":sum[j]/count}
            res = es.index(index='news_classification_'+model,doc_type=model,id=t,body=e)
            print("Res:",res['result'])
            t = t+1
            print("Average batch accuracy:",(sum[j]/count))
            time.sleep(5);
        else:
            time.sleep(1);
        j = j+1
