import pandas as pd

__author__ = "Ugur Turhal and Mario Tachikawa"
__date__ = "11.11.2023"
__license__ = "MIT"
"""
This class cleanes the CSV data that we have 
for the movie
"""


class Cleaner:

    def get_csv(self) -> None:
        csv = pd.read_csv("movies.csv",low_memory=False)
        self.read_csv(csv)

    def read_csv(self, csv: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(csv)
        self.clean_and_export(df)

    def clean_and_export(self, df: pd.DataFrame) -> None:
        a = ['gross_revenue_us_canada', 'X', 'mpaa_rating', 'X_id.x', 'X_id.y', 'image_url', 'estimated_budget',
             'opening_weekend_domestic_date', 'opening_weekend_domestic_gross', 'worldwide_gross', 'image_url_big']
        df = df.drop(axis=1, labels=a)
        df.to_csv("movies_clean.csv")
        print(list(df))

cleaner = Cleaner()
cleaner.get_csv()
