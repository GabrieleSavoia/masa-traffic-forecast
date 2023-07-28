from pyspark import keyword_only  
from pyspark.sql import SparkSession
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import time
from datetime import datetime
from pyspark.sql.window import Window

from pyspark.sql import functions as f
from pyspark.ml import Pipeline
from pyspark.ml.feature import  VectorAssembler


class DfResampler(Transformer):
    """
    Si occupa di eseguire il resampling del DataFrame Spark in funzione del livello di granularità specificato.
    """

    to_date = Param(Params._dummy(), 'to_date', 'Data limite di resample.', 
                       typeConverter=TypeConverters.toString)
    nan = Param(Params._dummy(), 'nan', 'Valore associato ad osservazioni nulle dopo il resampling', 
                       typeConverter=TypeConverters.toString)
    granularity = Param(Params._dummy(), 'granularity', 'Valore di granularità necessario per il resampling', 
                       typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, to_date=None, nan=None, granularity=None):
        super(DfResampler, self).__init__()
        self.to_date = Param(self, 'to_date', '')
        self.nan = Param(self, 'nan', '')
        self.granularity = Param(self, 'granularity', '')

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, to_date=None, nan=None, granularity=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def set_to_date(self, value):
        return self._set(to_date=value)

    def get_to_date(self):
        return self.getOrDefault(self.to_date)

    def get_nan(self):
        return self.getOrDefault(self.nan)

    def get_granularity(self):
        return self.getOrDefault(self.granularity)

    def transform(self, dataframe):
        to_date = self.get_to_date()
        nan = self.get_nan()
        granularity = self.get_granularity()

        spark = SparkSession \
            .builder \
            .getOrCreate()

        epoch = (f.col("ds").cast("bigint"))
        with_epoch = dataframe.withColumn("epoch", epoch)

        max_epoch = time.mktime(datetime.strptime(to_date, '%Y-%m-%d %H:%M:%S').timetuple())
        min_epoch = with_epoch.select(f.min("epoch")).collect()[0][0]

        range_date = spark.range(min_epoch, 
                                 max_epoch + 1, 
                                 granularity).toDF("epoch")

        df_resampled = range_date.join(with_epoch, "epoch", "left")\
                                .orderBy("epoch")\
                                .withColumn("ds_resampled", f.col("epoch").cast("timestamp"))\
                                .select(f.col('ds_resampled').alias('ds'), 
                                        f.col('y'))

        if nan is not None:
            df_resampled = df_resampled.na.fill(nan)

        return df_resampled


class TimeInfoExtractor(Transformer, HasInputCol, HasOutputCol, 
                        DefaultParamsReadable, DefaultParamsWritable):
    """
    Estrazione delle informazioni (ora, weekend) dalla colonna della data.
    """

    input_col = Param(Params._dummy(), 'input_col', 'Nome colonna di input da trasformare', 
                      typeConverter=TypeConverters.toString)
    info_to_extract = Param(Params._dummy(), 'info_to_extract', 'Informazioni da estrapolare dalla variabile del tempo',
                      typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, input_col=None, info_to_extract=None):
        super(TimeInfoExtractor, self).__init__()
        self._setDefault(input_col='ds', info_to_extract=['hour', 'weekend'])

        kwargs = self._input_kwargs
        self.set_params(**kwargs)
        
    @keyword_only
    def set_params(self, input_col=None, info_to_extract=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def get_input_col(self):
        return self.getOrDefault(self.input_col)
    
    def get_info_to_extract(self):
        return self.getOrDefault(self.info_to_extract)
    
    def transform(self, dataframe):
        input_col = self.get_input_col()
        info_to_extract = self.get_info_to_extract()

        if 'hour' in info_to_extract:
            dataframe = dataframe.withColumn('hour', f.hour(f.col('ds')))

        if 'weekend' in info_to_extract:
            dataframe = dataframe.withColumn('week_day', f.dayofweek(f.col("ds")))
            dataframe = dataframe.withColumn('weekend', f.when( (f.col('week_day') == 7) | (f.col('week_day') == 1), 1).otherwise(0))
            dataframe = dataframe.drop(f.col("week_day"))
        return dataframe


class LagExtractor(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Vengono calcolati i valori laggati della serie temporale.
    Viene ritornato in output un dataframe con un numero di colonne di lag pari al valore 'lag'.
    """

    lag = Param(Params._dummy(), 'lag', 'Lag da considerare.', 
                typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, lag=None):
        super(LagExtractor, self).__init__()
        self.lag = Param(self, 'lag', '')
        self._setDefault(lag=24)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, lag=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def get_lag(self):
        return self.getOrDefault(self.lag)

    def transform(self, dataframe):
        lag = self.get_lag()

        windowspec = Window.orderBy('ds')
        for h in range(1, lag+1):
            dataframe = dataframe.withColumn('lag_'+str(h),
                                            f.lag(f.col('y'), h).over(windowspec))


        return dataframe


class MovingAverageExtractor(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Ritorna per ogni osservazione, la media delle n osservazioni passate.
    """
    n = Param(Params._dummy(), 'n', 'Media degli n valori passati', 
                typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, n=None):
        super(MovingAverageExtractor, self).__init__()
        self.n = Param(self, 'n', '')
        self._setDefault(n=12)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, n=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def get_n(self):
        return self.getOrDefault(self.n)

    def transform(self, dataframe):
        n = self.get_n()

        windowspec = Window.orderBy('ds').rowsBetween(-n, -1)
        dataframe = dataframe.withColumn('avg_'+str(n), 
                                         f.avg('y').over(windowspec))

        return dataframe


class RemoveNan(Transformer):
    """
    Elimina le righe con almeno un valore nullo.
    """

    def transform(self, dataframe):
        return dataframe.na.drop()


class MakeLog(Transformer):
    """
    Esegue il logaritmo alla colonna 'y'.
    """

    def transform(self, dataframe):
        return dataframe.withColumn('y', f.log1p(f.col('y')))
