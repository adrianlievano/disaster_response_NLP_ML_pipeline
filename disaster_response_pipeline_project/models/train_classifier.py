import sys
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath):
    '''
    ARGUMENTS: a filepath to a SQL database
    OUTPUTS: this function returns the features, target column, and category 
    names of the datatable.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')

    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.loc[:, df.columns != 'message'].drop(['id', 'genre', 'original'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names

def tokenize(text):
    '''
    ARGUMENTS: text data
    OUTPUTS: a cleaned list of words that capture the features of the text input.
    The function removes stop_words in the english language and normalizes the try:
    after lemmatization. 
    '''
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    ARGUMENTS: 

    OUTPUTS: Prepares a scikit learn transformation and evaluation pipeline
    to process the text data. 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [5, 10, 15],
        'clf__estimator__min_samples_split': [2, 3, 4, 10]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    ARGUMENTS: Receives a model, X_test, Y_test, and a set of category names.

    OUTPUTS: Tables that summarize the precision, recall, f1-score for each 
    trained classifier on each category in category_names. 
    '''
    y_pred = model.predict(X_test)


    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    for column in category_names:
        print('\n---- {} ----\n{}\n'.format(column, classification_report(Y_test[column], y_pred_df[column])))

def save_model(model, model_filepath):
    '''
    ARGUMENTS:
    Receives a trained model and designated filepath to save. 

    OUTPUTS:
    A saved model accessible at the model_filepath
    '''
    joblib.dump(model, model_filepath)


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
