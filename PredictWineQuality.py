import sys
import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler                    
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def predictquality(classifier):
    if classifier == 'gbt':
        model = GBTRegressionModel.load('GBTml-model')
        print("Predicting with radient-Boosted Trees Model")
    elif classifier == 'rfc':
        model = RandomForestClassificationModel.load('Rfcml-model')
        print("Predicting with Random Forest classfier model")
    elif classifier == 'lgr':
        model = LogisticRegressionModel.load('lgreg-model')
        print("Predicting with logistic regression Model")
    elif classifier == 'dst':
        model = DecisionTreeClassificationModel.load('DSTml-model')
        print("Predicting with Decission tree Model")
    else:
        print("Entered wrong classifier\n")
        print("Enter lgr for logistic regression modelling\n gbt for Gradient-Boosted Trees  regressor \n dst for Decission tree classifier\n rfc for Randomforest classfier")  
        print("Predicting with Decission tree Model")
        model = DecisionTreeClassificationModel.load('DSTml-model')
    #predictions = model.transform(testdf)    
    #return predictions

 
sqlContext = SparkSession.builder.appName('PredictWinequality').getOrCreate()
 

testclassifier="dst"
if len(sys.argv) == 3:
  testFile = sys.argv[1]
  testclassifier=sys.argv[2]
  print("Arguments passed",len(sys.argv)," ",sys.argv[1]," ",sys.argv[2]) 
elif len(sys.argv) == 2:
  testFile = sys.argv[1]
  print("Arguments passed",len(sys.argv)," ",sys.argv[1],)
elif len(sys.argv) == 1:
  testFile = "TestDataset.csv"
else:
  print("wrong arguments entered")
  
try:      
  TestdataFrame = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(testFile)
except:

  print("\n not able to load file",testFile)  
try:
  TestdatafeatureColumns = [c for c in TestdataFrame.columns if c != 'quality']
  
  # create and configure the assembler for validation data
  assembler2 = VectorAssembler(inputCols=TestdatafeatureColumns, 
                         outputCol="features")
  TestdataDF = assembler2.transform(TestdataFrame)
except:

  print("\n df issue")

try:
  model = DecisionTreeClassificationModel.load('DSTml-model')
except:
  print("\n model issue")

try:
  predictquality(testclassifier)
  qualityPredictions=model.transform(TestdataDF)
  
  labelPredict = (qualityPredictions.select('""""quality"""""','prediction')).toPandas()
   
  print("Label and prediction\n",labelPredict.head())
  print("Scores for Predicting data\n============================================")
  print("Accuracy" , accuracy_score(labelPredict['""""quality"""""'], labelPredict['prediction']))
  TestF1score = f1_score(labelPredict['""""quality"""""'], labelPredict['prediction'], average='micro')
  print("F1 score:",TestF1score)
  print("Confusion Matrix:",confusion_matrix(labelPredict['""""quality"""""'], labelPredict['prediction']))
  
except:

  print("\n ***********************************************************************")
  print(" Test File cannot be found/File not passed. Please try again with following parameters\n")
  print(" Execute file using any of below: \n")
  print(" -----------------------------------------------------------------\n")
  print(" $ python3 PredictWineQuality.py <TestData_file_name>.csv \n")
  print(" $ python3 PredictWineQuality.py <TestData_file_name>.csv lgr\n")
  print(" $ python3 PredictWineQuality.py <TestData_file_name>.csv rfc\n")
  print(" $ python3 PredictWineQuality.py <TestData_file_name>.csv dst\n")
  print(" $ python3 PredictWineQuality.py <TestData_file_name>.csv gbt\n")
  print(" \n lgr for logistic regression modelling\n gbt for Gradient-Boosted Trees  regressor \n dst for Decission tree classifier\n rfc for Randomforest classfier\n")
  print(" -----------------------------------------------------------------\n")
  print(" --> Make sure the CSV as well as Model file for prediction is present in the same folder\n")
  print(" ***********************************************************************\n")
  exit()
 
