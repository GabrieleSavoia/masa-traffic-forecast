import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as f

def predictions_to_df(predictions, real):
    """
    Da un vettore di valori ad un DataFrame Pandas con le date corrsipondenti a quelle del DataFrame 'real' nella colonna 'ds'.
    """
    if not real.index.is_all_dates:
        real = real.set_index('ds')

    return pd.DataFrame(np.array(predictions), index=real.index, columns=['yhat'])

def visualize_data(y, yhat, name_plot='', x_name='', y_name='', title=''):
    """
    Avviene il salvataggio del grafico delle previsioni yhat rispetto ai valori reali y.

    y: valori reali (può essere un DataFrame Pandas con le colonne 'ds' e 'y' o un DataFrame Spark con le stesse colonne)
    yhat: valori previsti (può essere un DataFrame Pandas con le colonne 'ds' e 'y' oppure vettore di valori senza data)
    name_plot: path di salvataggio dell'immagine del grafico (.png viene aggiunto in automatico)
    x_name: nome asse x
    y_name: nome asse y
    title: titolo del grafico
    """

    plt.figure(figsize=(20,3))

    # Se non è un DataFrame Pandas è un DataFrame Spark e quindi lo converto in Pandas
    if not isinstance(y, pd.DataFrame):
        y = y.select(f.col('ds'), f.col('y')).toPandas().set_index('ds')

    # Se non è un DataFrame Pandas è un vettore di valori e quindi lo converto in un DataFrame Pandas
    if not isinstance(yhat, pd.DataFrame):
        yhat = predictions_to_df(yhat, y)

    plt.plot(y, label='real')
    plt.plot(yhat, label='prediction')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend()

    plt.savefig(name_plot+'.png')
    plt.close('all')
