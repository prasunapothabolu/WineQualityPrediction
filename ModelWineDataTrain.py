import sys
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
from sklearn.metrics import f1_score
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyspark.ml.classification import LogisticRegression
import time
import datetime
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import GBTRegressor


#conf = (SparkConf().setAppName("Trainwinequality"))
#sc = SparkContext("local", conf=conf)
#sc.setLogLevel("ERROR")
#sqlContext = SQLContext(s
sqlContext = SparkSession.builder.appName('TrainWineApp').getOrCreate()



if len(sys.argv) == 2:
	filepath = str(sys.argv[1])
	validatedataFrame = sqlContext.read.options(header='true', inferschema='true', sep=';').csv(filepath, header=True, inferSchema=True)
	print("***********************************************************************")
	print ("Argument passed is :", str(sys.argv[1]))
	print("***********************************************************************")
else:
	validatedataFrame = sqlContext.read.options(header='true', inferschema='true', sep=';').csv('ValidationDataset.csv', header=True, inferSchema=True)

TrainingdataFrame = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('TrainingDataset.csv')

TrainingdataFrame.printSchema()


tdtotcols = TrainingdataFrame.columns
vdtotcols =validatedataFrame.columns
# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in TrainingdataFrame.columns if c != 'quality']

# create and configure the assembler trining data
assembler = VectorAssembler(inputCols=featureColumns, 
                       outputCol="features")
# transform the original data
traindataDF = assembler.transform(TrainingdataFrame)
ValdatafeatureColumns = [c for c in TrainingdataFrame.columns if c != 'quality']

# create and configure the assembler for validation data
assembler2 = VectorAssembler(inputCols=ValdatafeatureColumns, 
                       outputCol="features")
ValiddataDF = assembler2.transform(validatedataFrame)
# logistic regression

lgreg = LogisticRegression( labelCol = tdtotcols[-1], featuresCol="features", maxIter=100, regParam=0.001, elasticNetParam=1, standardization=True)
lgregstart = time.time()

# Fit the data to the model
lrregrModelgen = lgreg.fit(traindataDF)

lgregend = time.time()

lgregtimetaken=lgregend-lgregstart
#print("Time taken for logistic regression ",lgregtimetaken)
lgrefpredictionsall = lrregrModelgen.evaluate(ValiddataDF)
lglabelPredict = (lgrefpredictionsall.predictions.select('""""quality"""""','prediction')).toPandas()
#type(lglabelPredict)
lrregrModelgen.write().overwrite().save("lgreg-model")

# Random forest  classfifier
RFC = RandomForestClassifier(labelCol = tdtotcols[-1], featuresCol="features", numTrees=60, maxBins=32, maxDepth=4, seed=42)

RFCstart = time.time()

RFCmodel=RFC.fit(traindataDF)

RFCend = time.time()

RFCtimetaken=RFCend-RFCstart
RFCpredictionsall = RFCmodel.transform(ValiddataDF)
RFClabelPredict = (RFCpredictionsall.select('""""quality"""""','prediction')).toPandas()
RFCmodel.write().overwrite().save('Rfcml-model')

# Decission tree classfier
DST = DecisionTreeClassifier(labelCol = tdtotcols[-1], featuresCol="features",  maxBins=32, maxDepth=4)

DSTstart = time.time()

DSTmodel=DST.fit(traindataDF)

DSTend = time.time()

DSTtimetaken=DSTend-DSTstart
DSTpredictionsall = DSTmodel.transform(ValiddataDF)
DSTlabelPredict = (DSTpredictionsall.select('""""quality"""""','prediction')).toPandas()
DSTmodel.write().overwrite().save('DSTml-model')


#Gradient Boost regressor
GBT = GBTRegressor(featuresCol='features',
                    labelCol = tdtotcols[-1],
                    maxIter=100,
                    maxDepth=5,
                    subsamplingRate=0.5,
                    stepSize=0.001)
GBTstart = time.time()

GBTmodel=GBT.fit(traindataDF)

GBTend = time.time()

GBTtimetaken=GBTend-GBTstart
GBTpredictionsall = GBTmodel.transform(ValiddataDF)
GBTlabelPredict = (GBTpredictionsall.select('""""quality"""""','prediction')).toPandas()
GBTmodel.write().overwrite().save('GBTml-model')

print("Time taken for logistic regression modelling in seconds",lgregtimetaken)
print("Scores of Modelling data using logistic regression\n================================================")
print("Accuracy:",lgrefpredictionsall.accuracy)
print("F1 score:",lgrefpredictionsall.weightedFMeasure())
# Overall statistics
F1score = f1_score(lglabelPredict['""""quality"""""'], lglabelPredict['prediction'], average='micro')
#print("F1- score: ", F1score)
print("Confusion Matrix:",confusion_matrix(lglabelPredict['""""quality"""""'], lglabelPredict['prediction']))
print("\n\nTime taken for Random Forest classfier modelling  in seconds",RFCtimetaken)
print("Scores of Modelling data using RandomForest Classifier\n=============================================")
print("Accuracy:",accuracy_score(RFClabelPredict['""""quality"""""'], RFClabelPredict['prediction']))
RFCF1score = f1_score(RFClabelPredict['""""quality"""""'], RFClabelPredict['prediction'], average='micro')
print("F1 score:",RFCF1score)
print("Confusion Matrix:",confusion_matrix(RFClabelPredict['""""quality"""""'], RFClabelPredict['prediction']))


print("\n\nTime taken for Gradient-Boosted Trees  regressor  modelling  in seconds",GBTtimetaken)
print("Scores of Modelling data using Decission tree classifier\n============================================")
print("Accuracy" , accuracy_score(GBTlabelPredict['""""quality"""""'], GBTlabelPredict['prediction']))
GBTF1score = f1_score(GBTlabelPredict['""""quality"""""'], GBTlabelPredict['prediction'], average='micro')
print("F1 score:",GBTF1score)
print("Confusion Matrix:",confusion_matrix(GBTlabelPredict['""""quality"""""'], GBTlabelPredict['prediction']))


print("\n\nTime taken for Decission tree classifier modelling  in seconds",DSTtimetaken)
print("Scores of Modelling data using Decission tree classifier\n============================================")
print("Accuracy" , accuracy_score(DSTlabelPredict['""""quality"""""'], DSTlabelPredict['prediction']))
DSTF1score = f1_score(DSTlabelPredict['""""quality"""""'], DSTlabelPredict['prediction'], average='micro')
print("F1 score:",DSTF1score)
print("Confusion Matrix:",confusion_matrix(DSTlabelPredict['""""quality"""""'], DSTlabelPredict['prediction']))




