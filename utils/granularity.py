from pyspark.sql import SparkSession, SQLContext
import os
from datetime import datetime, time
import pyspark.sql.functions as f
import pandas as pd
from datetime import datetime
import time

class ManagerGranularity():
    """
    Classe per la gestione delle granularità.
    Sono salvati come attributi di istanza tutti i valori che risultano 
    essere in funzione della granularità dei dati.
    """

    def __init__(self, granularity):

        if granularity == 'one_minute':
            self.pandas_freq = '1min'
            self.prophet_freq = 'min'
            self.spark_epoch = 60
        elif granularity == 'fifteen_minute':
            self.pandas_freq = '15min'
            self.prophet_freq = 'min'
            self.spark_epoch = 60 * 15
        elif granularity == 'one_hour':
            self.pandas_freq = '1h'
            self.prophet_freq = 'h'
            self.spark_epoch = 60 * 60