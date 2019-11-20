import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    ARGUMENTS: 
    messages_filepath: a string filepath to a csv file of text data
    categories_filepath: a string filepath to a csv file of message categories 
    for the disaster response data.

    OUTPUTS: 
    df: a dataframe containing the merged result of two dataframes on the foreign
    key of 'id.'
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df, how = 'left', on = 'id')
    return df

def clean_data(df):
    '''
    ARGUMENTS:
    df: a merged dataframe of messages and categories from the disaster response
    dataset. 

    OUTPUTS:
    clean_df: a preprocessed dataframe with appropriately formatted columns for 
    each category.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = []
    for col in row.unique():
        category_colnames.append(col[0:len(col)-2])

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype('str').str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    #Replace categories column in df with new category columns
    df = df.drop(columns=['categories'], axis=1)
    df = pd.concat([df, categories], axis= 1)

    #Drop any duplicates
    cleaned_df = df.drop_duplicates()

    return cleaned_df

def save_data(df, database_filename):
    '''
    ARGUMENTS:
    df: a dataframe
    database_filename: a filepath saving the dataframe into a sql tabe.

    OUTPUTS:
    '''
    engine = create_engine(database_filename)
    df.to_sql('InsertTableName', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
