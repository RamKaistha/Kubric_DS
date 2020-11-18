import requests
import pandas as pd
import scipy
import numpy
import csv
from numpy import genfromtxt

from sklearn.linear_model import LinearRegression


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area):
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response1=requests.get('https://storage.googleapis.com/kubric-hiring/linreg_test.csv')
    dc1= response1.content.decode('utf-8')
    cr1 = csv.reader(dc1.splitlines(), delimiter=',')
    cr1=pd.DataFrame(data=cr1)
    cr1=cr1.T
    cr1 = cr1[0].iloc[1:]
    cr1=cr1.to_numpy()

    cr1=cr1.reshape(-1,1)

    response = requests.get('https://storage.googleapis.com/kubric-hiring/linreg_train.csv')
    reg=LinearRegression()
    decoded_content = response.content.decode('utf-8')
    
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    cr=pd.DataFrame(data=cr)
    cr=cr.T
    cr = cr.iloc[1:]
    X=cr[0]
    Y=cr[1]
    X=X.to_numpy()
    Y=Y.to_numpy()
    X=X.reshape(-1,1)
    Y=Y.reshape(-1,1)
    reg.fit(X,Y)
    print(cr1)
    return(reg.predict(cr1))
    



if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
