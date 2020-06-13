# Data Scientist - Nanodegree

## Disaster Response Pipeline Project


### Introduction

The aim of this project  to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline will be developed to categorize these events which can be sent as messages to an appropriate disaster relief agency. A web app will also be developed where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


### Prerequisites

Following are the main libraries utilized:

* `nltk` 
* `sqlalchemy` 
* `Sklearn`


### ETL Pipeline

File _data/process_data.py_ contains following functions:

- Load the datsets - `messages.csv` and `categories.csv` 
- Merge the two datasets and derive the categories
- Clean the data
- Store the final cleaned dataste in a **SQLite database**


### ML Pipeline

File _models/train_classifier.py_ contains following functions:

- Load data from the **SQLite database**
- Split the data into train and test sets
- Build a text processing and machine learning pipeline
- Train and tunes a model using GridSearchCV
- Evaluate result on the test set
- Exports the final model as a pickle file


### Web App

Run the following commands in the project's root directory to set up your database and model.

 - To run ETL pipeline that cleans data and stores in database python:
   data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 

 - To run ML pipeline that trains classifier and saves python 
   models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

Run the following command in the app's directory to run your web app 
python run.py

Go to http://0.0.0.0:3001/

