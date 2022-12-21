# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the data from the csv files into a dataframe

    :param messages_filepath: csv file path for the data of the messages
    :type messages_filepath: str
    :param categories_filepath: csv file path for the data of the categories
    :type categories_filepath: str
    :return: dataframe with the two inputs merged by id
    :rtype: pd.DataFrame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # retrun the merged datasets
    return pd.merge(messages, categories, on="id")


def clean_data(df):
    """Clean the categories from the input, set appropriate column names, clean the categories classification and drop
    duplicates

    :param df: dataframe that holds the categories and messages
    :type df: pd.DataFrame
    :return: clean dataframe with an appropriate classification
    :rtype: pd.DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.slice(stop=-2)
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # changes the 2 in the related column to 1
    categories['related'] = categories['related'].astype('str').str.replace('2', '1')
    categories['related'] = categories['related'].astype('int')
    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df_new = pd.concat([df, categories], axis=1, sort=False)

    # drop duplicates
    df_no2 = df_new[~df_new.duplicated()]
    return df_no2


def save_data(df, database_filename):
    """Save the data to a sql database

    :param df: dataframe that will be saved in the given database
    :type df: pd.DataFrame
    :param database_filename: name of the database with correct ending (.db)
    :type database_filename: str
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', con=engine, if_exists='replace', index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()