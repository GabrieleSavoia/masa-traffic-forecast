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
from utils.visualizerData import visualize_data

"""
Esecuzione :
$ python classic_models.py

Valutazione dei modelli Arima e Prophet rispettivamente alle serie temporali 
A, B e C in funzione dei diversi livelli di aggregazione temporale.
In questo caso NON sonno utilizzate le Pandas Function API dal momento che si vogliono salvare i grafici delle previsioni.

Il tutto viene salvato nella directory 'test/ml/..'
"""

if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('Test modelli classici') \
        .getOrCreate()


    def valuate(model, model_name='', test_dir='', granularity=[], forecast=[], id_time_serie={}, from_date='', to_date=''):

        for gran in granularity:

            df = handleData.read_aggregation(spark, granularity=gran)
            df_ml = df.filter((f.col('name') == id_time_serie['name']) & 
                            (f.col('unique_id') == id_time_serie['unique_id']) &
                            (f.col('id_object') == id_time_serie['id_object'])
                            )\
                    .select(f.col('ds'), f.col('y'))\
                    .orderBy('ds').toPandas()
            
            for type_forecast in forecast:

                folder_name = test_dir + gran + '/'
                path_file = folder_name + type_forecast

                model_settings = {'from_date': from_date, 
                                  'to_date': to_date,
                                  'mode': 'local',
                                  'granularity': gran,
                                  'type_forecast': type_forecast,
                                  'log': False,
                                 }
                try: 
                    os.makedirs(folder_name) 
                except OSError as error: 
                    pass

                fn_model = model(**model_settings)
                total_metrics, y, yhat = fn_model(df_ml)

                # Salvataggio grafico
                visualize_data(y, yhat, 
                               name_plot=path_file + '_display', 
                               title='Previsione traffico '+model_name, 
                               y_name='densità traffico'
                              )
            
                # file valutazioni
                f_file = open(path_file + '_metrics.txt', 'w+')
                for metric, val in total_metrics.items():
                    f_file.write(metric+' : '+str(val)+' \n')
                f_file.close()




    test_dir = 'test/classic/'
    granularity = ['one_hour', 'fifteen_minute']
    forecast = ['1_step', '1_hour', '12_hour']
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

    models_to_test = {'arima': timeSerieSpark.Arima_TimeSerieSpark,
                      'prophet': timeSerieSpark.Prophet_TimeSerieSpark
                     }

    for id_time_serie in id_time_series:
        test_dir_ = test_dir+id_time_serie['name'].replace(' ', '')+'_'+str(id_time_serie['unique_id'])+'_'+str(id_time_serie['id_object'])+'/'
        for model_name, model in models_to_test.items():
            test_dir_model = test_dir_ + model_name+'/'
            valuate(model, model_name=model_name, 
                        test_dir=test_dir_model, 
                        granularity=granularity, 
                        forecast=forecast,
                        id_time_serie=id_time_serie, 
                        from_date=from_date, 
                        to_date=to_date
                    )




