# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### The Data ###
The goal of this natural language processing application is to read, process, and predict the needs of people that send disaster messages to an organization. In this dataset, there are 36 possible categories that messages can be classified as. With higher performance of classification, we can quickly direct messages to the appropriate aid organization so that the user can get the help they need fast. 

#### Imbalanced Data: How this Impacts the Results ####

The NLP multi-classification algorithm is trained on 36 message categories. It is important to consider that this is an imbalanaced classification algorithm, so the model will be skewed to predict a disaster response message to be  'aid_related' or 'weather_related' as opposed to 'child_alone' distress calls or a problem due to 'electricity' issues.  

![Bar Chart of Message Categories](https://github.com/adrianlievano/disaster_response_NLP_ML_pipeline/blob/master/disaster_response_pipeline_project/data_distribution.png "Title")

![Message Statistics](https://github.com/adrianlievano/disaster_response_NLP_ML_pipeline/blob/master/disaster_response_pipeline_project/data_statistics.png "Title")


### Heroku Hosting ###

You can find this application at:


### Next Steps: Forming Partnerships ###

Given the different categories from the training set and that most of the disaster related messages are due to:
    <ol>
        <li>aid_related</li>
        <li>weather_related</li>
        <li>direct_report</li>
        <li>other_aid</li> 
        <li>food</li>
    </ol>
   
We recommend that this organization partners with non-profits that can address a wide variety of needs in a short period of time. These include organizations like the American Red Cross, which has international reach, and ACTS World Relief, which unites, trains, and equips responders in these emergency situations. 
