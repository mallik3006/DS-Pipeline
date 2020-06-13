import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from pickle import dump
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    Input: Database name
    Returns: Text colums which is the independent feature and 
             36 categories which are the class features
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM MessageTable", engine)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    
    return X, y, df.iloc[:, 4:].columns


def tokenize(text):
    '''
    Tokenizes the text message
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds the model using a pipeline with the following esitmators:
        - CountVectorizer
        - TfidfTransformer
        - MultiOutputClassifier(RandomForestClassifier)
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
                    ])
    
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                   'tfidf__use_idf': (True, False),
                   'clf__estimator__min_samples_split': [2, 3],
                 }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model and prints the classification_report by category
    '''
    y_pred = model.predict(X_test)
    print("Labels:", np.unique(y_pred))
    print("Accuracy:", (y_pred == Y_test).mean())
    for c in range(y_pred.shape[1]):
        print("Category:", category_names[c])
        print("Accuracy:", (y_pred[:, c] == Y_test[:, c]).mean())
        print(classification_report(Y_test[:, c], y_pred[:, c], labels=np.unique(y_pred)))
        

def save_model(model, model_filepath):
    '''
    Save the final model to a pickle fil
    '''
    dump(model, open(model_filepath, 'wb'))        
    

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