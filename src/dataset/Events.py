import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Events:
    def __init__(self, id: str,
                        events_path: str = None,
                        select_path: str = None,
                        events_df: pd.DataFrame = None,
                        select_df: pd.DataFrame = None,
                        units: str = 'seconds',
                        column: str = 'text') -> None:
        self.id = id
        self.label_encoder = LabelEncoder()
        self.column = column
        if events_path is not None:
            self.events_path = events_path
            self.dataframe = pd.read_csv(self.events_path)
            self.select_path = select_path
            if self.select_path is not None:
                self.dataframe_select_from = pd.read_csv(self.select_path)
                #self.dataframe  = self.dataframe[self.dataframe['text'].str.contains('|'.join(self.dataframe_select_from['text'].values))]
                self.dataframe = self.dataframe[self.dataframe[column].isin(self.dataframe_select_from[column].values)]
            self.update_label_encoder()
        else:
            if events_df is not None:
                self.dataframe = events_df
                if select_df is not None:
                    self.dataframe_select_from = select_df
                    self.dataframe = self.dataframe[self.dataframe[column].isin(self.dataframe_select_from[column].values)]
                self.update_label_encoder()
        self.units = units

    def update_label_encoder(self):
        self.labels = self.dataframe[self.column].values
        self.data = self.label_encoder.fit_transform(self.dataframe[self.column])
        self.classes = self.label_encoder.classes_
