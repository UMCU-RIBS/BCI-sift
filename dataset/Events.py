# -------------------------------------------------------------
# BCI-sift
# Copyright (c) 2025
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Events:
    """
    Events class to load and store behavioral events for a given dataset. 

    Parameters:
    -----------
    :param id: str
        Name of subject or alias
    :param events_path: str
        Path to the csv file containing the events information. The csv file needs to contain at least the columns 'xmin' and 'text', 
        which represent the starting time point of each event and the event label, respectively. If the csv file contains a column 'duration', 
        the duration of each event is determined by this column.
    :param select_path: str
        Path to the csv file containing the events to select from the events csv file. The csv file needs to contain at least the column 'text', 
        which represents the event label. Only events with labels that are present in this csv file will be selected from the events csv file.
    :param events_df: pd.DataFrame
        Dataframe containing the events information. The dataframe needs to contain at least the columns 'xmin' and 'text', which represent the 
        starting time point of each event and the event label, respectively
    :param select_df: pd.DataFrame
        Dataframe containing the events to select from the events dataframe. The dataframe needs to contain at least the column 'text', 
        which represents the event label. Only events with labels that are present in this dataframe will be selected from the events dataframe.
    :param units: str
        Units of the time information in the events csv file or dataframe. The units can be either 'seconds' or 'samples'. If the units are 'seconds', 
        the time information will be converted to samples using the sampling rate of the dataset. If the units are 'samples', the time information will be used as is.
    :param column: str
        Name of the column in the events csv file or dataframe that contains the event labels. The default value is 'text'.
    
    Methods:
    --------
    - update_label_encoder: Update the label encoder with the event labels. 

    Returns:
    --------
    :return: None
    """
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
