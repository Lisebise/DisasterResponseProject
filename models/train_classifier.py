# import statements for needed packages
import sys
import pandas as pd
import re
import numpy as np
import sklearn
import pickle
import nltk

nltk.download(["punkt", "wordnet", "omw-1.4"])

from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df["message"].values
    # drop the not needed columns for y
    df.drop(columns=["message", "id", "original", "genre"], inplace=True)
    Y = df.values

    return X, Y, df.columns


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        # 'vect__ngram_range':((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [5],
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1, verbose=3)
    # running it with gridsearch takes a long time
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    print("Classification Report")
    for i, y_true in enumerate(Y_test[0]):
        print(classification_report(Y_test[:][i], y_pred[:][i], labels=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()