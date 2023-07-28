import numpy as np
from pmdarima.model_selection import RollingForecastCV

def mape(y, yhat): 
    """
    Mean Absolute Percentage Error
    Non è presente nelle librerie per il calcolo delle metriche.

    y : array dei valori di y reali
    yhat : array dei valori di y previsti

    return : min_mape, mape, max_mape
    """

    y, yhat = np.array(y), np.array(yhat)

    errors_vector = np.abs((y - yhat) / y)
    low = (min(errors_vector)*100)
    high = (max(errors_vector)*100)
    mape_metric = np.mean(np.abs(errors_vector)) * 100
    return low, mape_metric, high 


class RollingOriginForecastCV(RollingForecastCV):
    """
    Cross-Validation on a rolling origin (basis).

    Implementa '_iter_train_test_indices', ovvero un generatore Python che ad ogni iterazione 
    si comporta nel seguente modo :

    train, train, train, TEST,                          --> 1 iter
    train, train, train, train, TEST                    --> 2 iter
    train, train, train, train, train, TEST             --> 3 iter 
    train, train, train, train, train, train, TEST      --> 4 iter    

    PARAMETRI
    h : grandezza del numero di osservazioni di test (1 nel caso di esempio)
    step : quante osservazioni saltare per l'iterazione successiva (1 nel caso di esempio)
    initial : numero di osservazioni per il train da cui partire
    """

    def _iter_train_test_indices(self, y, X):
        """
            Esegue un yield dell'insieme di train e di test ad ogni iterazione 
        """
        n_samples = y.shape[0]
        initial = self.initial
        step = self.step
        h = self.h

        if initial is not None:
            if initial < 1:
                raise ValueError("Initial training size must be a positive "
                                    "integer")
            elif initial + h > n_samples:
                raise ValueError("The initial training size + forecasting "
                                    "horizon would exceed the length of the "
                                    "given timeseries!")
        else:
            initial = max(1, n_samples // 3)

        all_indices = np.arange(n_samples)
        window_start = 0
        while True:
            window_end = window_start + initial
            if window_end + h > n_samples:
                break

            train_indices = all_indices[:window_end]
            test_indices = all_indices[window_end: window_end + h]
            window_start += step

            yield train_indices, test_indices


