from pyspark.sql import SparkSession, SQLContext
import os
from datetime import datetime, time
import pyspark.sql.functions as f
import pandas as pd
from datetime import datetime
import time
from utils import handleData, transformerSpark, timeSerieSpark
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, FloatType

import json

from datetime import datetime

"""
Esecuzione :
$ python pandas_function_api.py

Esecuzione delle Pandas Function API per i modelli Arima e Prophet in riferimento all'elaborazione di 5 serie temporali.
I risultati sono solamente printati in console senza essere salvati su file.
"""

if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('Test Pandas Function API') \
        .getOrCreate()


    ################### SETTINGS ###################

    granularity = 'one_hour'
    forecast = '1_step'
    name = 'Via Pico Della Mirandola'
    unique_id = [1, 2, 3, 4, 5]
    from_date = '2020-06-24 01:00:00'
    to_date = '2020-06-30 23:00:00'

    models_to_test = {'arima': timeSerieSpark.Arima_TimeSerieSpark,
                      'prophet': timeSerieSpark.Prophet_TimeSerieSpark
                     }

    udf_schema = StructType([
                                StructField('name', StringType(), True),
                                StructField('unique_id', StringType(), True),
                                StructField('id_object', IntegerType(), True),
                                StructField('MAPE', FloatType(), True),
                                StructField('RMSE', FloatType(), True),
                                StructField('MSE', FloatType(), True),
                                StructField('MAE', FloatType(), True),
                            ])

    df = handleData.read_aggregation(spark, granularity=granularity)
    df = df.filter((f.col('name') == name) & (f.col('id_object') == 6)).orderBy('ds')
    df = df.where(f.col('unique_id').isin(unique_id))

    ################## Pandas Function API per i modelli Arima e Prophet #####################

    for model_name, model in models_to_test.items():
        model_settings = {'from_date': from_date, 
                        'to_date': to_date,
                        'mode': 'spark',
                        'granularity': granularity,
                        'type_forecast': forecast,
                        'log': False,
                        }

        fn_model = model(**model_settings)

        # Il risultato di fn_model deve seguire lo schema specificato in 'udf_schema'
        df_res= df.groupBy(['name', 'unique_id', 'id_object'])\
                    .applyInPandas(fn_model, schema=udf_schema)

        df_res.show()