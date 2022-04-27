
# Wine Quality Prediction ML model in Spark over AWS EMR, EC2

A parallel Machine Learning application in Amazon AWS cloud platform built to predict the quality of Wine using Apache Spark, python.
The Modeling data application is designed to run on EMR with 4 instances.
Prediction application will run on EC2 single instance(ubuntu) (with Docker and without Docker).

The Training Data was modelled using multiple classifiers to find classifier that give better performance.
Logistic regression, GBT regressor, Random forest classifier, Decision Tress classifier are used.
Decision Tree classifier give better performance. The metrics used to estimate performance are Run time,F1 score and Accuracy score.

The application is designed to run logistic regression, Random forest classifier, Decision tree classifier, GBT regressor. you can pass parameter to choose classifier.
if no classifier parameter is passed , application will predict using decision tree classifier since based on metrics it worked best.

Time taken for Decision tree classifier modelling  in seconds 0.8110044002532959

Scores of modelling data using Decision tree classifier : Accuracy is 100 and F1 score is 100



> Github Link: https://github.com/prasunapothabolu/WineQualityPrediction

> Docker Hub  Predict app container link : https://hub.docker.com/repository/docker/prasunapothabolu/mlwinequalpredict
 
   Docker Hub  train app container link: https://hub.docker.com/repository/docker/prasunapothabolu/winetrainapp
 
--- 
## Table of Contents

- [Upload Input Files to S3](#Upload-Input-Files-S3)
- [EMR Cluster Creation and Setup](#EMR-Installation)
- [Run Model Train data application on EMR Cluster](#Model-App-run)
- [EC2 Instance without Docker](#EC2-Instance-without-Docker)
- [EC2 Instance With Docker](#EC2-Instance-With-Docker)
- [Result and Summary](#Result&Summary)
- [TroubleShooting](#Troubleshooting)

---

### Upload Input Files to S3
1. Create S3 bucket mlprojectfiles at AWS

2. Login to AWS console and create a IAM role for ec2 instance to give access to s3 so that ec2 instace can have access to download (CSV files) and upload files (Model file) to s3.

3. Alternatively, we can give s3 bucket to public to give access to your files, **Note: This is generally not recommended as it has high security risks. It can be used when we dont have access to change IAM role policies.


4. Upload Datasets TrainingDataset.csv, ValidationDataset.csv and trained model data* to S3 Bucket

5. check files upload by accessing S3 bucket or executing below command at  aws cli

          aws s3 ls s3://mlprojectfiles/
          
          ***S3 URL***: ```
          url : s3://mlprojectfiles/ ```

---

### EMR Cluster Creation and Setup

1) **Creating an EMR cluster**

***Step 1:*** In the AWS dashboard under the `analytics` section click `EMR`

***Step 2:*** Now Click `Create Cluster` 

***Step 3:*** In the `General Configuration` for `Cluster Name` type cluster name.

* **Step 3.1**: Under ``Software configuration` in the applications field click the button for option `Spark: Spark 2.4.8 on Hadoop 2.10.1 YARN and Zeppelin 0.10.0``. 
          
* **Step 3.2:** Under `Hardware Configuration` leave the default `m5.xlarge` as the default m5.xlarge 
          
* **Step 3.3:**  Under `Hardware Configuration` , enter `4` instances for `Number of instances` 

* **Step 3.e:**  Under `Hardware Configuration` , unchecked "Enable auto termination" of Auto-termination field
          
* **Step 3.5:** Under `Security and access` select the EC2 key pair already created else create a new one
          
***Step 4:*** Click Create Cluster button. Wait for around 15 minutes for the cluster to start functioning. 

          ***Alternative Step***: If you want to create EMR cluster using command line interface , please use below

          ```
          aws emr create-cluster \
          --name "<My First EMR Cluster>" \
          --release-label <emr-5.35.0> \
          --applications Name=Spark \
          --ec2-attributes KeyName=<myEMRKeyPairName> \
          --instance-type m5.xlarge \
          --instance-count 4 \
          --use-default-roles	
          ```
***Step 5:***  Once cluster is stared and waiting. At "Security and access" ,click on Master server and edit inbound rules to add SSH type access
 
 
 ***Step 6:*** At summary section of EMR cluster, click on Connect to the Master Node Using SSH"  to find server details and commands to connect EMR cluster
 
---

### Run Model Train data application on EMR Cluster

***Step 1:*** connect master host using key and tool ( you can get master host details at EMR cluster summary seciton)

***Step 2:*** copy ModelWineDataTrain.py to master host

***Step 3:*** run below command to train model

spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 ModelWineDataTrain.py

***Step 4: Outputs files are loaded in output folder.
---
### EC2 Instance without Docker

1) **Creating EC2 Instance**
* **Step 1:** Under Compute Column in the `AWS Management Console` Click `EC2`
* **Step 2:** Under the `Instances` click `Launch Instance`
* **Step 3:** Select the `AMI` of your choice. `Ubuntu server 20.04` is usually preferred
* **Step 4:** Select `Instance Type` I've chosen t2.micro as I'm using AWS Educate and this gives me t2.micro under free tier elgible
* **Step 5:** Select Key pair
* Here one can either `review and launch` or `tweak security, configuration and storage` features of EC2.
* Launch EC2 Instance

2) **Installing Spark on EC2 Instance**
* **Step 1:** Update EC2 using the command 
```bash
sudo apt -y update
```
* **step 2:** Check python version ```python3 --version```
* **Step 3:** Instal pip 
```bash
sudo pip3 install --upgrade pip
```
* **step 4:** Install Java and check if it works
```bash
sudo apt install default-jre
java --version
```
* **step 5:** Install Py4j
```bash
pip install py4j
```
* **Step 6:** Install Spark and hadoop
```bash
!wget http://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz
!tar -zxvf spark-3.0.0-bin-hadoop2.7.tgz
```

* **step 7:** Install pyspark
```bash
 pip install pyspark
```
* **step 8:** Install numpy
```bash
sudo pip3 install numpy
```
* **step 9:** Install pandas
```bash
sudo pip3 install pandas
```
* **step 9:** Install scikit-learn
```bash
sudo pip3 install -U scikit-learn
```
 **step 10:** set environment variables
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

 export SPARK_HOME=/home/ubuntu/spark-3.0.0-bin-hadoop2.7
 
 export PATH=$PATH:$SPARK_HOME/bin
 
 export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
 
 export PYSPARK_PYTHON=python3
 
 export PATH=$PATH:$JAVA_HOME/jre/bin
 
3) **Running your Application in EC2**
* Copy the ModelWineDataTrain.py,PredictWineQuality.py,TrainingDataset.csv,ValidationDataset.csv files to the Ec2 instance ```
Note: we can copy using tools( mobaXtern, putty etc)
or using commands 
scp -i <"your .pem file"> PredictWineQuality.py :~/PredictWineQuality.py```

* Run the following command in Ec2 instance to start the model training if model file not exists on EC2 :
  Example of using input file as argument . since we set environment variables we can use command python3 directly
```bash
python3 ModelWineDataTrain.py validationDataset.csv
```
* Run the following command in Ec2 instance to start the predict :

Example of using input file as argument. since we set environment variables, we can use command python3 directly. a) the application is designed to run logistic regression, Random forest classifier, Decision tree classifier, GBT regressor. you can pass parameter to choose classifier. if no classifier parameter is passed, application will predict using decision tree classifier
      lgr for logistic regression

      rfc for Random forest classifier

      dst for Decision tree classifier

      gbt for GBT regressor

  b)you can pass test data file with any name. please make sure file exists on EC2.
  
commands:
python3 PredictWineQuality.py TestDataset.csv lgr

or

python3 PredictWineQuality.py TestDataset.csv rfc

or

python3 PredictWineQuality.py TestDataset.csv dst

or

python3 PredictWineQuality.py TestDataset.csv gbt


* Outputs files are loaded in output folder.

### EC2 Instance With Docker

1) **Installation**
> Assuming that the above steps (#EC2-Instance-without-Docker) were clear for the setting up of EC2. Go ahead with the below steps post the setting up and running of EC2 to install docker
* **Step 1:** Command for installing the most recent Docker Community Edition package.
```bash
sudo apt install docker.io
```
* **Step 2:** Start the Docker service.
```bash
sudo service docker start
```
* **Step 3:**  check docker status
```bash
systemctl status docker.service
```
* **Step 4:** Add the ubuntu to the docker group so you can execute Docker commands without using sudo.
```bash
sudo usermod -a -G docker ubuntu
```
* **Step 4:** Verify that the ubuntu can run Docker commands without sudo.
```bash
docker  --version or docker info
```

2) **Building Dockerfile**
* **Step 1:** Type `touch Dockerfile` to create a Dockerfile ( at location where code files ,data files exists)
* **Step 2:** edit Dockerfile with all commands(which install all libraries,software etc) to create the Dockerfile Image to automate the process
* **Step 3:** 
```bash
sudo docker build . -f Dockerfile -t <Image name of your choice>

ex:sudo docker build . -f Dockerfile -t winequaltrain

and check image created or not using command sudo docker images
```

3) **Pushing and Pulling created Image to DockerHub**
* **Step 1:** Login to your dockerhub account through ec2
```bash
docker login: Type your credentials

if any issue please run below command sudo chmod 666 /var/run/docker.sock
```
* **Step 2:** In order to push docker type the following commands
```bash
docker tag <Local Ec2 Repository name>:<Tag name> <dockerhub username>/<local Ec2 Repository name>

ex:

```
```bash
docker push <dockerhub username>/<local Ec2 Repository name>

ex:

docker push prasunapothabolu/winetrainapp

```
* **Step 3:** Pulling your Dockerimage back to Ec2 
```bash
docker pull <dockerhub username>/<Repository name>:<tag name>
```
Example:
```bash
docker pull prasunapothabolu/winetrainapp:latest
```
* **Step 4:** Running my dockerimage
```bash
sudo docker run -t <user> <Given Image name>
```
Example:
```bash
sudo docker run -v /home/ubuntu prasunapothabolu/winetrainapp TrainingDataset.csv

```
*** Repeating same steps for prediction file image creation

changed Dockerfile to use PredictWineQuality.py app

sudo docker build . -f Dockerfile -t mlwinequalpredict

docker tag mlwinequalpredict prasunapothabolu/mlwinequalpredict

docker push prasunapothabolu/mlwinequalpredict

docker pull prasunapothabolu/mlwinequalpredict:latest

sudo docker run -v /home/ubuntu prasunapothabolu/mlwinequalpredict TestDataset.csv

---
### Result & Summary
> Decision tree classifier worked well compare to other classfiers . It ran in less time with accuracy 100 and F1 Score 100
* Outputs files are loaded in output folder.
---
### Troubleshooting

> In a scenario where one is unable to execute the program in the desired format then in such a case run the below commands

> These commands can be used for checking installation errors

``` bash 
sudo pip install --upgrade pip
```
```bash 
sudo pip3 install -U scikit-learn
```
```bash 
sudo pip install pyspark --no-cache-dir
```
```bash 
sudo pip install findspark
```
```bash 
sudo pip install numpy
```
```bash 
sudo apt install python3-pip
```
```bash 
sudo docker rmi -f image image_name
```
```bash 
sudo docker images
```
```bash 
sudo docker system prune -a
```
```bash 
sudo docker ps
```
```bash 
sudo docker ps -a
```
```bash 
sudo docker run -t image_tag_name
```
```bash 
sudo docker build . -f docker_file_name -t image_tag_name
```
```bash 
sudo docker start container_name
```
```bash 
sudo docker stop container_name
```
```bash 
sudo docker login -u sampathgonnuru <ur docker-hub username>
```
```bash 
sudo groupadd docker
```

```bash 
sudo chmod 666 /var/run/docker.sock
```
```bash 
sudo docker images
```
```bash 
below commands remove and cleanup all images
docker image rm -f <Image id>
docker system prune --all --force
```

