
# Wine Quality Prediction ML model in Spark over AWS EMR, EC2

A parallel Machine Learning application in Amazon AWS cloud platform built to predict the quality of Wine using Apache Spark, python.
The Modeling data application is designed to run on EMR with 4 instances.
Prediction application will run on EC2 single instance (with Docker and without Docker).

The Training Data was modelled using multiple classifiers to find classifier that give better performance.
Logistic regression, GBT regressor, Random forest classifier, Decision Tress classifier are used.
Decision Tree classifier give better performance. The metrics used to estimate performance are Run time,F1 score and Accuracy score.

> Github Link: https://github.com/prasunapothabolu/WineQualityPrediction

> Docker Hub:
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
***Step 5:***  Once cluser is stared and waiting. At "Security and access" ,click on Master server and edit inbound rules to add SSH type access
 
 
 ***Step 6:*** At summary section of EMR cluster, click on Connect to the Master Node Using SSH"  to find server details and commands to connect EMR cluster
 
---
###Run Model Train data application on EMR Cluster



---
### EC2 Instance without Docker

1) **Creating EC2 Instance**
* **Step 1:** Under Compute Column in the `AWS Management Console` Click `EC2`
* **Step 2:** Under the `Instances` click `Create Instance`
* **Step 3:** Select the `AMI` of your choice. `Amazon Linux 2 AMI` is usually preferred
* **Step 4:** Select `Instance Type` I've chosen t2.micro as I'm using AWS Educate and this gives me t2.micro under free tier elgible
* **Step 5:** Here one can either `review and launch` or `tweak security, configuration and storage` features of EC2.
* Launch EC2 Instance

2) **Installing Spark on EC2 Instance**
* **Step 1:** Update EC2 using the command 
```bash
sudo yum update -y
```
* **step 2:** Check python version ```python3 --version```
* **Step 3:** Instal pip 
```bash
sudo pip3 install --upgrade pip
```
* **step 4:** Install Java and check if it works
```bash
sudo yum install java-1.8.0-devel
java --version
```
* **step 5:** Install Py4j
```bash
pip install py4j
```
* **Step 6:** Install Spark and hadoop
```bash
!wget https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop2.7.tgz    
!tar xf spark-3.2.1-bin-hadoop2.7.tgz 
```

* **step 7:** Install findspark
```bash
sudo pip3 install findspark
```
* **step 8:** Install findspark
```bash
sudo pip3 install numpy
```
3) **Running your Application in EC2**
* Copy the PredictWineQuality.py file to the Ec2 instance ```
Note: we can copy using tools( mobaXtern, putty etc)
scp -i <"your .pem file"> predict.py :~/predict.py```

* Run the following command in Ec2 instance to start the model prediction :
Example of using S3 file as argument 
```bash
spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 predict.py s3://mywineproject/ValidationDataset.csv
```
---

### EC2 Instance With Docker

1) **Installation**
> Assuming that the above steps (#EC2-Instance-without-Docker) were clear for the setting up of EC2. Go ahead with the below steps post the setting up and running of EC2 to install docker
* **Step 1:** Command for installing the most recent Docker Community Edition package.
```bash
sudo yum install docker -y
```
* **Step 2:** Start the Docker service.
```bash
sudo service docker start
```
* **Step 3:**  Add the ec2-user to the docker group so you can execute Docker commands without using sudo.
```bash
sudo usermod -a -G docker ec2-user
```
* **Step 4:** Verify that the ec2-user can run Docker commands without sudo.
```bash
docker  --version or docker info
```

2) **Building Dockerfile**
* **Step 1:** Type `touch Dockerfile` to create a Dockerfile
* **Step 2:** nano Dockerfile and create the Dockerfile Image to automate the process
* **Step 3:** 
```bash
sudo docker build . -f Dockerfile -t <Image name of your choice>
```

3) **Pushing and Pulling created Image to DockerHub**
* **Step 1:** Login to your dockerhub account through ec2
```bash
docker login: Type your credentials
```
* **Step 2:** In order to push docker type the following commands
```bash
docker tag <Local Ec2 Repository name>:<Tag name> <dockerhub username>/<local Ec2 Repository name>
```
```bash
docker push <dockerhub username>/<local Ec2 Repository name>
```
* **Step 3:** Pulling your Dockerimage back to Ec2 
```bash
docker pull <dockerhub username>/<Repository name>:<tag name>
```
Example:
```bash
docker pull sampathgonnuru/cs643-project2:latest
```
* **Step 4:** Running my dockerimage
```bash
sudo docker run -t <Given Image name>
```
Example
```bash
docker run -it sampathgonnuru/cs643-project2:latest s3//mywineproject/ValidationDataset.csv 
```
---
### Result & Summary


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



