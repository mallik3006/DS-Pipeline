import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input: takes the messgaes and catergories dataset
    Output:  Returns a merged dataframe with 36 categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df


def clean_data(df):
    '''
    Input: takes the merged dataframe
    Output: Returns a cleaned dataframe 
          - Remove duplicates
          - Drop records with NaN 
          - Exlcude records with labels other than 0 or 1
    '''
    df = df.drop_duplicates()
    df = df.dropna(subset=df.iloc[:, 4:].columns, how='any')
    df = df.loc[df['related'] != 2]
    
    return  df


def save_data(df, database_filename):
    '''
    Save the cleaned dataframe to a sqlite table
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageTable', engine, index=False)
    
    
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