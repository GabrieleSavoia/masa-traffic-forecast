import time
from datetime import datetime
import pandas as pd
import numpy as np
from utils import evaluation as custom_eval
from utils.granularity import ManagerGranularity

from fbprophet.diagnostics import prophet_copy

from pmdarima.model_selection import RollingForecastCV

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

class PandasOperation():
    """
    Classe che si occupa di gestire le operazioni di preprocessing sul DataFrame Pandas.
    """

    def __init__(self, granularity):
        """
        granularity: valore di granularità della serie temporale
        """
        self.manager_granularity = ManagerGranularity(granularity)
        self.operation_done = {'log': False}

    @classmethod
    def extract_info(cls, df):
        """
        Metodo di classe che si occupa di ottenere le informazioni della serie temporale
        contenute nel DataFrame in cui sono presenti anche i dati delle osservazioni.
        Usato con le Pandas Function API.

        df: DataFrame Pandas iniziale

        return: dizionario con le informazioni, DataFrame con solo le colonne 'ds' e 'y'.
        """
        info =  {'name': df['name'].values[0],
                 'unique_id': df['unique_id'].values[0],
                 'id_object': df['id_object'].values[0],
                }
        df = df.drop(columns=['name', 'unique_id', 'id_object'])
        return info, df

    @classmethod
    def split_set(cls, df, train_size=0.8):
        """
        Avviene lo split in train e test set.

        df: DataFrame Pandas iniziale
        train_size: indica la grandenzza del train (0.8 --> 80%)

        return: ritorna il train e il test set
        """
        if df is None:
            return None, None

        train = df[:int(len(df)*train_size)]
        test = df[len(train):]
        return train, test
    
    def preprocess(self, df, to_date, from_date='', log=False):
        """
        Avvengono le operazioni di preprocessing dei dati:
        - resampling (da 'from_date' a 'to_date') in funzione del livello di granularità
        - gestione valori nulli
        - logaritmo (se specificato)

        df: DataFrame Pandas della serie temporale
        from_date: (compreso) se non specificato viene considerata la data minima
        to_date : (compreso) data finale da considerara eper il preprocessing
        log: se eseguire o no il logaritmo

        return DataFrame Pandas preprocessato
        """
        self.operation_done['log'] = log

        if from_date == '':
            from_date = df['ds'].min()

        df = df[(df['ds'] >= from_date) & (df['ds'] <= to_date)]
        date_interval = pd.DataFrame(pd.date_range(from_date, to_date, 
                                                   freq=self.manager_granularity.pandas_freq), 
                                     columns=['interval_date'])

        res_df = pd.merge(date_interval, df, left_on='interval_date', right_on='ds', how='outer')\
                   .drop(columns=['ds'])\
                   .rename(columns={'interval_date': 'ds'})

        res_df['y'] = res_df['y'].fillna(0.1)

        if log:
            res_df['y'] = np.log(res_df['y']) 
        
        return res_df

    def inverse_op(self, y):
        """
        Vengono eseguite l'inverso delle operazioni effettuate nel preprocessing 
        al vettore y. In questo caso si esegue l'esponenziale nel caso in cui il logaritmo sia stato calcolato nel preprocessing, ma è possibile estendere 
        le possibili trasformazioni.

        y: vettore delle osservazioni

        return: vettore y con trasformazioni inverse
        """
        y_np_array = np.array(y)

        if self.operation_done['log']:
            y_np_array = np.exp(y_np_array)

        return y_np_array

    def compute_metric(self, y, yhat):
        """
        Vengono ritornate le metriche di valutazione dopo aver calcolato l'inverso (se necessario) dei vettori.

        y: vettore dei valori reali
        yhat: vettore dei valori previsti

        return dizionario con le metriche MAPE, RMSE, MSE, MAE
        """
        y_inverse = self.inverse_op(y)
        yhat_inverse = self.inverse_op(yhat)

        low, mape, high  = custom_eval.mape(y=y_inverse, yhat=yhat_inverse)
        mse = mean_squared_error(y_inverse, yhat_inverse)
        rmse = sqrt(mse) 
        mae = mean_absolute_error(y_inverse, yhat_inverse)

        return {'low_percent_error': low,
                'MAPE': mape,
                'high_percent_error ': high,
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
               }

    def evaluate(self, df, model, type_model, type_forecast='1_step', initial_train_size=0.8, exog_handler=None):
        """
        Valutazione dei modelli tramite la metodologia di Cross-validation on a rolling origin (o basis).

        df: DataFrame Pandas della serie temporale 
        model: riferimento al modello da valutare
        type_model: 'arima' o 'prophet' (nome del modello) (parametro forse di tropppo in quanto si può ricavare dal modello)
        type_forecast: 1_step, 1_hour, 12_hour 
        initial_train_size: 0.8 --> (80% dei dati come train iniziale)
        exog_handler: riferimento al gestore delle variabili esogene di Arima
        """

        eval_conf = {'1_step': {'1min': {'h': 1, 'step': 1},
                                '15min': {'h': 1, 'step': 1},
                                '1h': {'h': 1, 'step': 1},
                            }, 
                    '1_hour': {'1min': {'h': 60, 'step': 60},
                               '15min': {'h': 4, 'step': 4},
                               '1h': {'h': 1, 'step': 1},
                            },
                    '12_hour': {'1min': {'h': 60*12, 'step': 60},
                                '15min': {'h': 4*12, 'step': 4},
                                '1h': {'h': 12, 'step': 1},
                            },
                    }
        
        conf = eval_conf[type_forecast][self.manager_granularity.pandas_freq]
        initial = int(len(df)*initial_train_size)

        df_ = df.copy().reset_index()
        cv = custom_eval.RollingOriginForecastCV(**conf, initial=initial)
        cv_gen = cv.split(df_)

        step_value = {'y': [], 'yhat':[]}
        dates = []

        for train, test in cv_gen:

            step_value['y'].append(list(df_.loc[test]['y']))
            train = df_.loc[train]
            test = df_.loc[test]
            if type_model=='arima':   
                train = train.set_index('ds')

                exog_train = exog_handler.get_exogenous_df(df_prediction=train)
                exog_test = exog_handler.get_exogenous_df(df_prediction=test)
                model_trained = model.fit(train, exogenous=exog_train)

                yhat = model_trained.predict(n_periods=conf['h'], 
                                             exogenous=exog_test,
                                            )
            elif type_model=='prophet':
                m = prophet_copy(model)
                m_trained = m.fit(train)
                yhat = m_trained.predict(test)['yhat']
                
            step_value['yhat'].append(list(yhat))
            dates.append(max(test['ds']))

        y = [sum(list_value) / len(list_value) for list_value in step_value['y']]
        yhat = [sum(list_value) / len(list_value) for list_value in step_value['yhat']]
        metrics = self.compute_metric(y, yhat)

        df_total = pd.DataFrame({'ds': dates, 'y': y, 'yhat': yhat})
        df_y = df_total[['ds', 'y']].set_index('ds')
        df_yhat = df_total[['ds', 'yhat']].set_index('ds')

        return metrics, df_y, df_yhat

                        

class Arima_exogenous():
    """
    Gestore delle variabili esogene di Arima.
    Necessità di una classe in modo da non dover passare ogni volta il tipo di variabili 
    esogene da considerare (orario, weekend ecc.)
    """

    def __init__(self, df, hour=True, week_day=False, weekend=False):
        """
        df: DataFrame Pandas della serie temporale
        hour: considerazione delle ore come variabile esogena
        week_day: considerazione del giorno della settimana come varibaile esogena
        weekend: considerazione del weekend come vribaile esogena
        """
        self.df = df
        self.hour = hour
        self.week_day = week_day
        self.weekend = weekend

    def get_exogenous_df(self, df_prediction=None):
        """
        Se prediction=None          --> variabili exog. calcolate in riferimento al DataFrame Pandas 'self.df'
        Se prediction=DataFrame     --> variabili exog. calcolate in rifeirmento al DataFrame 'prediction'
                                        E' il caso del calcolo delle var exog. per un DataFrame Pandas delle osservazioni di previsione (vedi valutazione)

        return: DataFrame Pandas contenente le variabili esogene ricavate.
        """

        if not self.hour and not self.week_day and not self.weekend:
            return None

        if df_prediction is not None:
            df = df_prediction.copy()
        else:
            df = self.df.copy()

        if not df.index.is_all_dates:
            df = df.set_index('ds')
        
        if self.hour:
            # One Hot Encoding Ora del giorno
            df['hourday'] = df.index.hour

            # https://stackoverflow.com/questions/37425961/dummy-variables-when-not-all-categories-are-present
            possible_hours = [h for h in range(0, 24)]
            hour_dummies = pd.get_dummies(df['hourday'])
            hour_dummies = hour_dummies.T.reindex(possible_hours).T.fillna(0).astype('int64')

            #hour_dummies.columns = ['hourday-'+ str(w) for w in range(0,24)]
            df = pd.concat([df, hour_dummies], axis=1).drop(['hourday'],axis=1)

        if self.weekend:
            # Weekend
            df['weekend'] = (df.index.dayofweek>4).astype(int)

        if self.week_day:
            # Giorno settimana
            df['week_day'] = (df.index.dayofweek).astype(int)
        
        if 'ds' in df:
            df = df.drop(columns=['ds'])

        if 'y' in df:
            return df.drop(['y'], axis=1)

        return df
