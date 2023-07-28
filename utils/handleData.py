from pyspark.sql import SparkSession, SQLContext
import os
from datetime import datetime, time
import pyspark.sql.functions as f
import pandas as pd
from datetime import datetime
import time

granularity_available = ['one_minute', 'fifteen_minute', 'one_hour', 'one_day']

def read_aggregation(spark, granularity='one_hour'):
    """
    Caricamento aggregazioni temporali create dallo streaming e salvate nella directory 'sink_big'.

    spark : sessione di Spark creata nel main
    granularity : valore di granularità dei dati che si vuole trattare

    return : DataFrame con la granularità specificata
    """
    if granularity not in granularity_available:
        print('Frequenza aggregazioni non valida')
        return False

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    df = sqlContext.read.parquet(os.path.join(os.getcwd(), 'sink_big', 'sink_'+granularity , '*'))\
                        .select(f.col('name'),
                                f.col('unique_id'),
                                f.col('id_object'), 
                                f.col('window_end').alias('ds'),
                                f.col('average_count').alias('y'))\
                        .orderBy('name', 'unique_id', 'id_object')

    return df

def select_time_series(df, min_observations):
    """
    Vengono selezionate solamente le serie temporali con un numero minimo (min_observations)
    di osservazioni.
    Funzione che può tornare utile in certi contesti ma non è usata nel progetto.

    df : DataFrame contenente un certo numero di serie temporali
    min_observations : numero minimo di osservazioni per ciascuna delle serie temporali

    return : DataFrame filtrato e numero di serie temporali presenti nel DataFrame
    """
    df_analisi = df.groupBy('name', 'unique_id', 'id_object').count().withColumnRenamed("count", "#timeserie_data")\
                   .orderBy('#timeserie_data', ascending=False)

    df_series_considered = df_analisi.filter(f.col('#timeserie_data') > min_observations).select('name', 'unique_id', 'id_object')
    df_prepared = df.join(df_series_considered, ['name', 'unique_id', 'id_object'], 'inner').orderBy('ds')

    total_time_series = df_prepared.groupBy(['name', 'unique_id', 'id_object']).count()\
                                   .orderBy('count', ascending=False)\
                                   .count()

    return df_prepared, total_time_series
