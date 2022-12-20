
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Next to the Anaconda distribution of Python you need the following packages/librariers to execute the project.
- pandas
- numpy
- sklearn
- nltk
- sqlalchemy
- sys
- re
- pickle

The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I used data from FigureEight that presents messages and categories from disaster responses. I was interested in applying natural lannguage processing and training a supervised machine learning model. Following that, the trained model is able to categorize new messages in the known categories. This helps to understand and filter important messages and therefor extraxt useful information in situation like disasters. The model is represented in a small web interface that allows the user to categorize tehir own message to test the model.

Instrcution to run it in a console
- To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- to run the run.py
	`cd app`; `python run.py`

## File Descriptions <a name="files"></a>

The work is separated into data, models and app. 
In the data folder, the needed input data disaster_categories.csv and disaster_messages.csv, as well as the process_data.py which extract, tranforms and load the data. It loads the data into a database DisasterResponse.db.
In the models folder, the train_classifier.py loads the cleaned data from the database and creates a modeled which is trained. It will then later extract the model als a pickle file (classifier.pkl).
In the app folder, the run.py creates an interface to show the counts of messaged per category and creates a user input for categorizing messaged. The needed html can be found in the templates folder.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to FigureEight for the data. Otherwise, feel free to use the code here as you would like! 

