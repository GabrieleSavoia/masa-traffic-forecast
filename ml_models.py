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
from utils.timeSerieSpark import ML_TimeSerieSpark
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
import time
from utils.visualizerData import visualize_data

"""
Esecuzione :
$ python ml_models.py

Valutazione dei modelli DecisionTree e GradientBoostedTree rispettivamente alle serie temporali 
A, B e C in funzione dei diversi livelli di aggregazione temporale.

Il tutto viene salvato nella directory 'test/ml/..'
"""


if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('Test modelli di machine learning') \
        .getOrCreate()


    def valuate(model, model_name='', test_dir='', granularity=[], id_time_serie={}, from_date='', to_date=''):

        for gran in granularity:

            df = handleData.read_aggregation(spark, granularity=gran)
            df_ml = df.filter((f.col('name') == id_time_serie['name']) & 
                            (f.col('unique_id') == id_time_serie['unique_id']) &
                            (f.col('id_object') == id_time_serie['id_object'])
                            )\
                    .select(f.col('ds'), f.col('y'))\
                    .orderBy('ds')
            
            try: 
                os.makedirs(test_dir) 
            except OSError as error: 
                pass

            ts_spark = ML_TimeSerieSpark(model, df_ml, gran)
            ts_spark.preprocess(from_date=from_date, 
                                to_date=to_date,
                                h_lags=2, avg_h=2, use_hour=True
                                )
            res = ts_spark.evaluate()

            name_test = gran
            f_file = open(test_dir+name_test+'.txt', 'w+')
            for metric, val in res['metrics'].items():
                f_file.write(metric+' : '+str(val)+' \n')
            f_file.close()

            visualize_data(res['test'], 
                            res['yhat'], 
                            name_plot=test_dir+'display_'+gran, 
                            title='Previsione traffico '+model_name, 
                            y_name='densità traffico'
                          )


    """
    Parametri per la valutazione :
    test_dir : nome della directory in cui salvare i risultati
    granularity : i vari livelli di granularità da utilizzare
    id_time_serie : lista di dizionari con le info di ciascuna serie temporale
    from_date, to_date : per ogni serie temporale elabora le osservazioni da 'from_date' a 'to_date'
    models_to_test : DecisionTree e GradientBoostedTree
    """

    test_dir = 'test/ml/'
    granularity = ['one_hour', 'fifteen_minute', 'one_minute']

    # A, B e C rispettivamente
    id_time_series = [
                      {'name': 'Via Pico Della Mirandola',
                       'unique_id': 1,
                       'id_object': 6,
                      },
                      {'name': 'Via Pico Della Mirandola',
                       'unique_id': 6,
                       'id_object': 6,
                      },
                      {'name': 'Stradello Soratore',
                       'unique_id': 1,
                       'id_object': 6,
                       }
                     ]
    
    from_date = '2020-06-24 01:00:00'
    to_date = '2020-06-30 23:00:00'

    models_to_test = {'decision_tree': DecisionTreeRegressor(featuresCol = 'features', labelCol='y'),
                      'gradient_boosted_tree': GBTRegressor(featuresCol = 'features', labelCol='y')
                     }

    for id_time_serie in id_time_series:
        test_dir_ = test_dir+id_time_serie['name'].replace(' ', '')+'_'+str(id_time_serie['unique_id'])+'_'+str(id_time_serie['id_object'])+'/'
        for model_name, model in models_to_test.items():
            test_dir_model = test_dir_ + model_name+'/'
            valuate(model, model_name=model_name, 
                        test_dir=test_dir_model, 
                        granularity=granularity, 
                        id_time_serie=id_time_serie, 
                        from_date=from_date, 
                        to_date=to_date
                    )
            




