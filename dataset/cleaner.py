import pandas as pd
import numpy as np

__author__ = "Ugur Turhal and Mario Tachikawa"
__date__ = "11.11.2023"
__license__ = "MIT"


class Cleaner:

    """
    This class cleans the CSV data that we have
    for the movie dataset. This dataset is then needed for our
    Chatbot. This have the following
    """

    def get_csv(self) -> None:
        """
        Reads the original File
        :return:
        """
        csv = pd.read_csv("movies.csv", low_memory=False)
        self.read_csv(csv)

    def read_csv(self, csv: pd.DataFrame):
        df = pd.DataFrame(csv)
        self.clean_and_export(df)

    def clean_and_export(self, df: pd.DataFrame) -> None:
        a = ['gross_revenue_us_canada', 'X', 'mpaa_rating', 'X_id.x', 'X_id.y', 'image_url', 'estimated_budget',
             'opening_weekend_domestic_date', 'opening_weekend_domestic_gross', 'worldwide_gross', 'image_url_big']
        df = df.drop(axis=1, labels=a)

        patterns_to_replace = {'I\)': 0, 'II\)': 0, 'IV\)': 0, 'V\)': 0, 'VI\)': 0, 'VIII': 0,
                               'XIII': 0, 'IX\)': 0, 'X\)': 0, 'XI\)': 0, 'XXII\)': 0,
                               'XXII': 0, 'XXIX': 0, 'XVII': 0, 'XXXV': 0}

        df = df.replace(patterns_to_replace, regex=True).fillna(0)
        df['year'] = df['year'].astype(int)
        # df = df[df['year'] >= 2000]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # ignore all movies without stars
        df = df[df['stars'] != 0]
        df['stars'] = df['stars'].apply(
            lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
        df = df.explode('stars')

        df['genre'] = df['genre'].apply(
            lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
        df = df.explode('genre')
        df['director'] = df['director'].apply(
            lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
        df = df.explode('director')
        df = df[df['popularity_score'] != 0]
        df.replace(['NA', np.NaN, '', None], 0, inplace=True)
        df.to_csv("movies_clean.csv")
        print(list(df))


cleaner = Cleaner()
cleaner.get_csv()
