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
    """Load data into the needed arrays for the mdoel, messages and categories

    :param database_filepath:
    :type database_filepath: str
    :return:
        - X - messages to categorize
        - Y - categories that are selected
        - df.columns - column names
    :type:
        - X - np.array
        - Y - np.array
        - df.columns - pd.Series
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df["message"].values
    # drop the not needed columns for y
    df.drop(columns=["message", "id", "original", "genre"], inplace=True)
    Y = df.values

    return X, Y, df.columns


def tokenize(text):
    """Separate the input text into word tokens and lemmatize them

    :param text: text that will be tokenized
    :type text: str
    :return clean_tokens: list with the word tokens
    :rtype: list
    """
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
    """Build model that can be fitted and later on predict values

    :return cv: grid search model with which the training can be done
    :rtype cv: GridSearchCV
    """
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
    """Evaluate the model based on the accuracy and the classification report

    :param model: sklearn model that can predict
    :param X_test: messgaes that will be categorized
    :param Y_test: categories of the X_test
    :param category_names: list containing all categories
    """
    # predict on test data
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    print("Classification Report")
    for i, y_true in enumerate(Y_test[0]):
        print(classification_report(Y_test[:][i], y_pred[:][i]))


def save_model(model, model_filepath):
    """Save the model as pickle file

    :param model: Trained model that will be saved
    :param model_filepath: string where the model will be saved
    """
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