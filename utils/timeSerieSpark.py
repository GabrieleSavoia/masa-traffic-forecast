from pyspark.sql import functions as f
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import datetime
import os
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, FloatType

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation as prophet_cv
from pyspark.sql import functions as f

from pmdarima.arima import auto_arima
from pmdarima.model_selection import RollingForecastCV
from pmdarima import model_selection

from utils.transformerSpark import DfResampler, TimeInfoExtractor, LagExtractor, MovingAverageExtractor, RemoveNan, MakeLog
from utils.transformerPandas import PandasOperation, Arima_exogenous
from utils import evaluation as custom_eval
from utils.granularity import ManagerGranularity


############################ MACHINE LEARNING ###################################

class ML_TimeSerieSpark():
    """
    Classe che si occupa di processare un dataframe di una serie temporale in un dataframe predisposto all'utilizzo 
    di un algoritmo di machine learning supervisionato.
    Viene poi valutato sul test set con le opportune metriche.
    Con i modelli di machiine learning sono considerate SOLO previsioni 1_step (no quelle 1_hour o 12_hour disponibili in Prophet e Arima).
    """

    def __init__(self, model, df, granularity, date_col='ds', label_col='y'):
        """
        model: modello di machine learning che si vuole usare per la previsione
        df: DataFrame Spark contenente i dati di una singola serie temporale
        granularity: livello di granularità della serie
        date_col: nome della colonna relativa alla data delle osservazioni
        label_col: nome della colonna relativa al valore delle osservazioni
        """
        self.model = model
        self.trained_model = None

        self.df = df
        self.date_col = date_col
        self.label_col = label_col

        self.granularity = ManagerGranularity(granularity)

        self.preprocessed = None

        # Operazioni fatte nel preprocessing di cui bisogna farne l'inverso nell' evaluation
        self.operation_done = {'log': False} 

    def preprocess(self, to_date, from_date='', h_lags=0, avg_h=0,
                   use_hour=False, use_weekend=False,
                   log=False):
        """
        Preprocessing e feature engineering del dataframe, in modo da adattarlo ad un problema di ml supervisionato.
        Vengono utilizzati i transformer creati appositamente per trattare le serie temporali.

        from_date : (compresa) 
        to_date : (compresa) avviene un resampling da 'from_date' a 'to_date'
        h_lags : grandezza della window da considerare
        avg_h : numero di osservazioni passate su cui calcolare la media
        use_hour : se utilizzare come feature l'orario
        use_weekend : se utilzzare come feature il weekend
        log : esegue il logaritmo alla colonna 'y'
        """
        self.operation_done['log'] = log

        stages = []
        stages.append(DfResampler(to_date=to_date, nan=0.1, granularity=self.granularity.spark_epoch))
        if use_hour or use_weekend:
            stages.append(TimeInfoExtractor())
        if log:
            stages.append(MakeLog())
        if h_lags:
            stages.append(LagExtractor(lag=h_lags))
        if avg_h:
            avg_col = ['avg_'+str(avg_h)]
            stages.append(MovingAverageExtractor(n=avg_h))
        else:
            avg_col = []
        
        stages.append(RemoveNan())

        lag_cols = ['lag_'+str(h) for h in range(1, h_lags+1)]

        time_info_cols = []
        if use_hour:
            time_info_cols.append('hour')
        if use_weekend:
            time_info_cols.append('weekend')

        stages.append(VectorAssembler(inputCols=[] + lag_cols + avg_col + time_info_cols,
                                      outputCol='features'))

        pipeline_fitted = Pipeline(stages=stages).fit(self.df)
        self.preprocessed = pipeline_fitted.transform(self.df).filter(f.col('ds') >= from_date).filter(f.col('ds') <= to_date)

    def train(self, multiplier=4):
        """
        Esegue il train del modello su tutti i dati processati.
        Parametro multiplier utilizzatio solo per fare alcune prove ma nel progetto si usa il valore di default (4)

        multiplier : fattore moltiplicativo proporzionale alla grandezza della serie temporale di train (1, 2, 3, 4)
                     1 --> 25% dei dati totali
                     2 --> 50% dei dati totali
                     3 --> 75% dei dati totali
                     4 --> 100% dei dati totali
        """
        train, _ = self.split_set(train_percent=multiplier*25)

        self.trained_model = self.model.fit(train)

    def split_set(self, train_percent=80):
        """
        Suddivisione dei dati processati in train e test set.

        train_percent: percentuale di dati considerati come train

        return dataframe di train e di test
        """
        df_ranked = self.preprocessed.withColumn('rank', 
                                                 f.percent_rank().over(Window.partitionBy().orderBy('ds')))
        train = df_ranked.where('rank <= '+str(train_percent/100)).drop('rank')
        test = df_ranked.where('rank > '+str(train_percent/100)).drop('rank')

        return train, test

    def evaluate(self, train_percent=80, pred_col='prediction'):
        """
        Avviene la valutazione del modello allenato secondo le metriche di RMSE, MSE, MAE, MAPE.
        Per questi modelli sono considerate solamente previsioni 1_step (no 1_hour o 12_hour come per Arima e Prophet).

        train percent : percentuale di suddivisione in train e test set
        pred_col : nome della colonna contenente i valori di previsione

        return : dizionario python contenente le metriche, i dataframe di train e test, e i valori di previsione
                 in modo tale da poter essere visualizzati graficamente da una apposita funzione.

        """
        train, test = self.split_set(train_percent)

        model_trained = self.model.fit(train)
        yhat = model_trained.transform(test)

        # Calcolo l'esponenziale se è stato fatto il logaritmo.
        if self.operation_done['log']:
            yhat = yhat.withColumn(pred_col, f.expm1(f.col(pred_col)))\
                       .withColumn(self.label_col, f.expm1(f.col(self.label_col)))

        evaluation = RegressionEvaluator(labelCol= 'y', 
                                         predictionCol= 'prediction', 
                                         metricName='rmse')
        evalutation_result = {}
        evalutation_result['rmse'] = evaluation.evaluate(yhat)
        evalutation_result['mse'] = evaluation.evaluate(yhat, {evaluation.metricName: "mse"})
        evalutation_result['mae'] = evaluation.evaluate(yhat, {evaluation.metricName: "mae"})

        y = np.array(yhat.select(f.col(self.label_col)).collect())
        yhat = np.array(yhat.select(f.col(pred_col)).collect())
        lowest_percent_error, mape_val, highest_percent_error = custom_eval.mape(y=y, yhat=yhat)
        evalutation_result['lowest_percent_error'] = lowest_percent_error
        evalutation_result['mape'] = mape_val
        evalutation_result['highest_percent_error'] = highest_percent_error

        return {'metrics': evalutation_result, 'train': train, 'test': test, 'yhat': yhat}


############################ PROPHET ###################################

def Prophet_TimeSerieSpark(to_date, 
                           from_date= '', 
                           mode='local',
                           granularity='one_hour',
                           type_forecast='1_step',
                           log=False):
    """
    Funzione che si occupa di configurare le variabili necessarie per la previsione con Prophet.
    Ritorna il riferimento alla funzione che esegue il modello Prophet.
    Se mode='spark' allora è predisposta per l'utilizzo delle Pandas Function API

    from_date: (compresa)
    to_date: (compresa), data finale del datatset (avviene un resampling da 'from_date' a 'to_date').
    mode: 'local' o 'spark', a seconda se si vuole eseguire il modello in locale o con le Pandas Function API
    granularity: valore di granularità della serie da prevedere
    type_forecast: '1_step', '1_hour', '12_hour' sono i possibili tipi di previsione
    log_data : (default=False) indica se utilizzare o no il logaritmo.

    return: riferimento alla funzione che esegue il modello Prophet
    """
    def fn_prophet(df):
        """
        Funzione che effettua le previsioni e calcola le metriche di errore del modello Prophet.

        df: dataframe Pandas contenente i dati di una singola serie temporale

        return: se mode='local' --> ritorna i valori delle metriche con i relativi valori reali e previsti da visualizzare su grafico
                se mode='spark' --> ritorna un DataFrame Pandas che segue esattamente lo schema specificato nella Pandas Function API
        """
        train_size = 0.80
        pandas_operation = PandasOperation(granularity)

        if mode == 'spark':
            info, df = PandasOperation.extract_info(df)
        df = pandas_operation.preprocess(df, to_date, from_date=from_date, log=log)

        df_train, df_test = PandasOperation.split_set(df, train_size=train_size)
        model = Prophet(
            daily_seasonality=True
        )
        model = model.fit(df_train)
        
        total_metrics, df_y, df_yhat = pandas_operation.evaluate(df, 
                                                                 model, 
                                                                 'prophet', 
                                                                 type_forecast=type_forecast,
                                                                 initial_train_size=train_size,
                                                                )

        if mode == 'spark':
            result = info
            result['MAPE'] = [total_metrics['MAPE']]
            result['RMSE'] = [total_metrics['RMSE']]
            result['MSE'] = [total_metrics['MSE']]
            result['MAE'] = [total_metrics['MAE']]   
            return pd.DataFrame(result)

        if mode == 'local':
            return total_metrics, df_y, df_yhat

    return fn_prophet


####################### ARIMA ########################

def Arima_TimeSerieSpark(to_date, 
                         from_date= '', 
                         mode='local',
                         granularity='one_hour',
                         type_forecast='1_step',
                         log=False):
    """
    Funzione che si occupa di configurare le variabili necessarie per la previsione con Arima.
    Ritorna il riferimento alla funzione che esegue il modello Arima.
    Se mode='spark' allora è predisposta per l'utilizzo delle Pandas Function API

    from_date: (compresa)
    to_date: (compresa), data finale del datatset (avviene un resampling da 'from_date' a 'to_date').
    mode: 'local' o 'spark', a seconda se si vuole eseguire il modello in locale o con le Pandas Function API
    granularity: valore di granularità della serie da prevedere
    type_forecast: '1_step', '1_hour', '12_hour' sono i possibili tipi di previsione
    log_data : (default=False) indica se utilizzare o no il logaritmo.

    return: riferimento alla funzione che esegue il modello Arima
    """

    def fn_arima(df):
        """
        Funzione che effettua le previsioni e calcola le metriche di errore del modello Arima.

        df: dataframe Pandas contenente i dati di una singola serie temporale

        return: se mode='local' --> ritorna i valori delle metriche con i relativi valori reali e previsti da visualizzare su grafico
                se mode='spark' --> ritorna un DataFrame Pandas che segue esattamente lo schema specificato nella Pandas Function API
        """
        train_size = 0.8
        pandas_operation = PandasOperation(granularity)

        if mode == 'spark':
            info, df = PandasOperation.extract_info(df)

        df = pandas_operation.preprocess(df, to_date, 
                                        from_date=from_date, 
                                        log=log,
                                        ).set_index('ds')

        exog_handler = Arima_exogenous(df, 
                                hour=True, 
                                week_day=True, 
                                weekend=True)
        exog = exog_handler.get_exogenous_df()

        df_train, df_test = PandasOperation.split_set(df, train_size=train_size)
        exog_train, exog_test = PandasOperation.split_set(exog, train_size=train_size)

        arima_model = auto_arima(df_train, seasonal=True, exogenous=exog_train)

        total_metrics, df_y, df_yhat = pandas_operation.evaluate(df, 
                                                                 arima_model, 
                                                                 'arima', 
                                                                 type_forecast=type_forecast,
                                                                 exog_handler=exog_handler,
                                                                 initial_train_size=train_size,
                                                                )

        if mode == 'spark':
            result = info
            result['MAPE'] = [total_metrics['MAPE']]
            result['RMSE'] = [total_metrics['RMSE']]
            result['MSE'] = [total_metrics['MSE']]
            result['MAE'] = [total_metrics['MAE']]   
            return pd.DataFrame(result)

        if mode == 'local':
            return total_metrics, df_y, df_yhat

    return fn_arima


