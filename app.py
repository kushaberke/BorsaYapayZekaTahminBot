import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = df.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(f'Predicted with the accuracy of : {r2_score(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        print(f'Predicted Closing Price For Day {day} is : {i}')
        day += 1


stock = "ASELS.IS"
today = datetime.date.today()
duration = 3000
before = today - datetime.timedelta(days=duration)
start_date = before
end_date = today-datetime.timedelta(days=2)
print(end_date)
scaler = StandardScaler()

data = download_data(stock,start_date,end_date)

num = 1

engine = LinearRegression()
model_engine(engine, num)
