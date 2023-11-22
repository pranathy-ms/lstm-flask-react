'''
Goal of LSTM microservice:
1. LSTM microservice will accept the GitHub data from Flask microservice and will forecast the data for next 1 year based on past 30 days
2. It will also plot three different graph (i.e.  "Model Loss", "LSTM Generated Data", "All Issues Data") using matplot lib 
3. This graph will be stored as image in Google Cloud Storage.
4. The image URL are then returned back to Flask microservice.
'''
# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client
client = storage.Client()

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    
    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    # Creating the image path for model loss, LSTM generated image and all issues data image


    CLOSED_ISSUES_FORECAST_IMAGE_NAME = "forecast_closed_issues_" + type + "_" + repo_name + ".png"
    CLOSED_ISSUES_FORECAST_URL = BASE_IMAGE_PATH + CLOSED_ISSUES_FORECAST_IMAGE_NAME 
   # Forecasting for created issues
    CREATED_ISSUES_FORECAST_IMAGE_NAME = "forecast_created_issues_" + type + "_" + repo_name + ".png"
    CREATED_ISSUES_FORECAST_URL = BASE_IMAGE_PATH + CREATED_ISSUES_FORECAST_IMAGE_NAME 

    PULLS_FORECAST_IMAGE_NAME = "forecast_pulls_" + type + "_" + repo_name + ".png"
    PULLS_FORECAST_URL = BASE_IMAGE_PATH + PULLS_FORECAST_IMAGE_NAME 

    COMMITS_FORECAST_IMAGE_NAME = "forecast_commits_" + type + "_" + repo_name + ".png"
    COMMITS_FORECAST_URL = BASE_IMAGE_PATH + COMMITS_FORECAST_IMAGE_NAME 

    BRANCHES_FORECAST_IMAGE_NAME = "forecast_branches_" + type + "_" + repo_name + ".png"
    BRANCHES_FORECAST_URL = BASE_IMAGE_PATH + BRANCHES_FORECAST_IMAGE_NAME 

    CONTRIBUTORS_FORECAST_IMAGE_NAME = "forecast_contributors_" + type + "_" + repo_name + ".png"
    CONTRIBUTORS_FORECAST_URL = BASE_IMAGE_PATH + CONTRIBUTORS_FORECAST_IMAGE_NAME 

    RELEASES_FORECAST_IMAGE_NAME = "forecast_releases_" + type + "_" + repo_name + ".png"
    RELEASES_FORECAST_URL = BASE_IMAGE_PATH + RELEASES_FORECAST_IMAGE_NAME 

    FORECAST_DAY_OF_WEEK_CREATED_IMAGE_NAME = "forecast_day_of_week_created_" + type + "_" + repo_name + ".png"
    FORECAST_DAY_OF_WEEK_CREATED_URL = BASE_IMAGE_PATH + FORECAST_DAY_OF_WEEK_CREATED_IMAGE_NAME

    FORECAST_DAY_OF_WEEK_CLOSED_IMAGE_NAME = "forecast_day_of_week_closed_" + type + "_" + repo_name + ".png"
    FORECAST_DAY_OF_WEEK_CLOSED_URL = BASE_IMAGE_PATH + FORECAST_DAY_OF_WEEK_CLOSED_IMAGE_NAME

    FORECAST_MONTH_CLOSED_IMAGE_NAME = "forecast_month_closed_" + type + "_" + repo_name + ".png"
    FORECAST_MONTH_CLOSED_URL = BASE_IMAGE_PATH + FORECAST_MONTH_CLOSED_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_day_of_week_created')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_DAY_OF_WEEK_CREATED_IMAGE_NAME)
    
    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_day_of_week_closed')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_DAY_OF_WEEK_CLOSED_IMAGE_NAME)

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_month_closed')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_MONTH_CLOSED_IMAGE_NAME)

   # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_created_issues')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + CREATED_ISSUES_FORECAST_IMAGE_NAME) 
    
    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_closed_issues')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + CLOSED_ISSUES_FORECAST_IMAGE_NAME) 
    
    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_pulls')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + PULLS_FORECAST_IMAGE_NAME)

   # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_commits')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + COMMITS_FORECAST_IMAGE_NAME) 
    
    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_branches')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + BRANCHES_FORECAST_IMAGE_NAME) 

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_contributors')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + CONTRIBUTORS_FORECAST_IMAGE_NAME)  

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('forecast_releases')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + RELEASES_FORECAST_IMAGE_NAME)  
    
    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    
    new_blob = bucket.blob(FORECAST_DAY_OF_WEEK_CREATED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_DAY_OF_WEEK_CREATED_IMAGE_NAME)
        
    new_blob = bucket.blob(FORECAST_DAY_OF_WEEK_CLOSED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_DAY_OF_WEEK_CLOSED_IMAGE_NAME)
        
    new_blob = bucket.blob(FORECAST_MONTH_CLOSED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_MONTH_CLOSED_IMAGE_NAME)
        
    new_blob = bucket.blob(CREATED_ISSUES_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CREATED_ISSUES_FORECAST_IMAGE_NAME)
        
    new_blob = bucket.blob(CLOSED_ISSUES_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CLOSED_ISSUES_FORECAST_IMAGE_NAME)
        
    new_blob = bucket.blob(PULLS_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + PULLS_FORECAST_IMAGE_NAME)    
        
    new_blob = bucket.blob(COMMITS_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + COMMITS_FORECAST_IMAGE_NAME)     
        
    new_blob = bucket.blob(BRANCHES_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + BRANCHES_FORECAST_IMAGE_NAME)    
        
    new_blob = bucket.blob(CONTRIBUTORS_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + CONTRIBUTORS_FORECAST_IMAGE_NAME)     

    new_blob = bucket.blob(RELEASES_FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + RELEASES_FORECAST_IMAGE_NAME)    
    # Construct the response
    json_response = {
        "forecast_day_of_week_created": FORECAST_DAY_OF_WEEK_CREATED_URL,
        "forecast_day_of_week_closed": FORECAST_DAY_OF_WEEK_CLOSED_URL,
        "forecast_month_closed": FORECAST_MONTH_CLOSED_URL,
        "forecast_created_issues": CREATED_ISSUES_FORECAST_URL,
        "forecast_closed_issues": CLOSED_ISSUES_FORECAST_URL,
        "forecast_pulls": PULLS_FORECAST_URL,
        "forecast_commits": COMMITS_FORECAST_URL,
        "forecast_branches": BRANCHES_FORECAST_URL,
        "forecast_contributors": CONTRIBUTORS_FORECAST_URL,
        "forecast_releases": RELEASES_FORECAST_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
