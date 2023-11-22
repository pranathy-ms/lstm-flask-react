
# Import all the required packages 
import os
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np


# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of  requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''
@app.route('/api/github', methods=['POST'])
def github():
    body = request.get_json()
    print(body)
    # Extract the choosen repositories from the request
    repo_name = body['repository']
    # Add your own GitHub Token to run it local
    token = os.environ.get(
        'GITHUB_TOKEN', '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')
    GITHUB_URL = [
        "https://api.github.com/repos/openai/openai-cookbook",
        "https://api.github.com/repos/openai/openai-cookbook",
        "https://api.github.com/repos/openai/openai-python",
        "https://api.github.com/repos/openai/openai-quickstart-python",
        "https://api.github.com/repos/milvus-io/pymilvus",
        "https://api.github.com/repos/SeleniumHQ/selenium",
        "https://api.github.com/repos/golang/go",
        "https://api.github.com/repos/google/go-github",
        "https://api.github.com/repos/angular/material",
        "https://api.github.com/repos/angular/angular-cli",
        "https://api.github.com/repos/SebastianM/angular-google-maps",
        "https://api.github.com/repos/d3/d3",
        "https://api.github.com/repos/facebook/react",
        "https://api.github.com/repos/tensorflow/tensorflow",
        "https://api.github.com/repos/keras-team/keras",
        "https://api.github.com/repos/pallets/flask"
    ]
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "open"
    }
    repository_url = GITHUB_URL + "repos/" + repo_name
    # Fetch GitHub data from GitHub API
    repository = requests.get(repository_url, headers=headers)
    # Convert the data obtained from GitHub API to JSON format
    repository = repository.json()

    today = date.today()

    issues_reponse = []
    # Iterating to get issues for every month for the past 2 months
    for i in range(2):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        types = 'type:issue'
        repo = 'repo:' + repo_name
        ranges = 'created:' + str(last_month) + '..' + str(today)
        # By default GitHub API returns only 30 results per page
        # The maximum number of results per page is 100
        # For more info, visit https://docs.github.com/en/rest/reference/repos 
        per_page = 'per_page=100'
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = types + ' ' + repo + ' ' + ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_issues = requests.get(query_url, headers=headers, params=params)
        # Convert the data obtained from GitHub API to JSON format
        search_issues = search_issues.json()
        issues_items = []
        try:
            # Extract "items" from search issues
            issues_items = search_issues.get("items")
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for issue in issues_items:
            label_name = []
            data = {}
            current_issue = issue
            # Get issue number
            data['issue_number'] = current_issue["number"]
            # Get created date of issue
            data['created_at'] = current_issue["created_at"][0:10]
            if current_issue["closed_at"] == None:
                data['closed_at'] = current_issue["closed_at"]
            else:
                # Get closed date of issue
                data['closed_at'] = current_issue["closed_at"][0:10]
            for label in current_issue["labels"]:
                # Get label name of issue
                label_name.append(label["name"])
            data['labels'] = label_name
            # It gives state of issue like closed or open
            data['State'] = current_issue["state"]
            # Get Author of issue
            data['Author'] = current_issue["user"]["login"]
            issues_reponse.append(data)

        today = last_month

    df = pd.DataFrame(issues_reponse)

    # Daily Created Issues
    df_created_at = df.groupby(['created_at'], as_index=False).count()
    dataFrameCreated = df_created_at[['created_at', 'issue_number']]
    dataFrameCreated.columns = ['date', 'count']
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])
    # Convert specified columns to appropriate data types
    columns_to_convert = ['issues', 'issues_closed', 'stars', 'forks']
    df[columns_to_convert] = df[columns_to_convert].astype(int)
    # Ensure 'repo' column is in string format (adjust if needed)
    df['repo'] = df['repo'].astype(str)
    
   # A Line Chart to plot the issues for every Repo
    repo_names = df['repo'].unique()
    plt.figure(figsize=(10, 6))
    for repo_name in repo_names:
        repo_data = df[df['repo'] == repo_name]
        plt.plot(repo_data['created_at'], repo_data['issues'], label=repo_name)

    plt.title('Issues Over Time for Every Repo')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.legend()
    plt.show()



    # A Bar Chart to plot the stars for every Repo
    plt.figure(figsize=(10, 6))
    for repo_name in repo_names:
        repo_data = df[df['repo'] == repo_name]
        plt.bar(repo_data['created_at'], repo_data['stars'], label=repo_name, alpha=0.7)

    plt.title('Stars for Every Repo Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Stars')
    plt.legend()
    plt.show()

    # A Bar Chart to plot the forks for every Repo
    plt.figure(figsize=(10, 6))
    for repo_name in repo_names:
        repo_data = df[df['repo'] == repo_name]
        plt.bar(repo_data['created_at'], repo_data['forks'], label=repo_name, alpha=0.7)

    plt.title('Forks for Every Repo Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Forks')
    plt.legend()
    plt.show()

    # A Bar Chart to plot the issues closed for every week for every Repo
    plt.figure(figsize=(12, 6))
    for repo_name in repo_names:
        repo_data = df[df['repo'] == repo_name]
        plt.bar(repo_data['closed_at'], repo_data['issues_closed'], label=repo_name, alpha=0.7)

    plt.title('Weekly Issues Closed for Every Repo')
    plt.xlabel('Week')
    plt.ylabel('Number of Issues Closed')
    plt.legend()
    plt.show()

    # A Stack-Bar Chart to plot the created and closed issues for every Repo
    plt.figure(figsize=(12, 6))
    for repo_name in repo_names:
        repo_data = df[df['repo'] == repo_name]
        plt.bar(repo_data['created_at'], repo_data['issues'], label='Created - ' + repo_name, alpha=0.7)
        plt.bar(repo_data['closed_at'], repo_data['issues_closed'], label='Closed - ' + repo_name, alpha=0.7)

    plt.title('Created and Closed Issues for Every Repo Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.legend()
    plt.show() 
        
    '''
    # Monthly Created Issues
    # Format the data by grouping the data by month
    # ''' 
    created_at = df['created_at']
    month_issue_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d')
    month_issue_created.index = month_issue_created.dt.to_period('m')
    month_issue_created = month_issue_created.groupby(level=0).size()
    month_issue_created = month_issue_created.reindex(pd.period_range(
    month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
    month_issue_created_dict = month_issue_created.to_dict()
    created_at_issues = []
    for key in month_issue_created_dict.keys():
        array = [str(key), month_issue_created_dict[key]]
        created_at_issues.append(array)

    # Weekly Closed Issues
    # Format the data by grouping the data by week
    closed_at = df['closed_at'].sort_values(ascending=True)
    week_issue_closed = pd.to_datetime(
        pd.Series(closed_at), format='%Y-%m-%d')
    week_issue_closed.index = week_issue_closed.dt.to_period('W-Mon')  # Group by week starting on Monday
    week_issue_closed = week_issue_closed.groupby(level=0).size()
    week_issue_closed = week_issue_closed.reindex(pd.period_range(
        week_issue_closed.index.min(), week_issue_closed.index.max(), freq='W-Mon'), fill_value=0)
    week_issue_closed_dict = week_issue_closed.to_dict()
    closed_at_issues = []
    for key in week_issue_closed_dict.keys():
        array = [str(key), week_issue_closed_dict[key]]
        closed_at_issues.append(array)


    '''
        1. Hit LSTM Microservice by passing issues_response as body
        2. LSTM Microservice will give a list of string containing image paths hosted on google cloud storage
        3. On recieving a valid response from LSTM Microservice, append the above json_response with the response from
            LSTM microservice
    '''
    created_at_body = {
        "issues": issues_reponse,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
    closed_at_body = {
        "issues": issues_reponse,
        "type": "closed_at",
        "repo": repo_name.split("/")[1]
    }

    # Update your Google cloud deployed LSTM app URL (NOTE: DO NOT REMOVE "/")
    LSTM_API_URL = "your_lstm_gcloud_url/" + "api/forecast"

    '''
    Trigger the LSTM microservice to forecasted the created issues
    The request body consists of created issues obtained from GitHub API in JSON format
    The response body consists of Google cloud storage path of the images generated by LSTM microservice
    '''
    created_at_response = requests.post(LSTM_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})
    
    '''
    Trigger the LSTM microservice to forecasted the closed issues
    The request body consists of closed issues obtained from GitHub API in JSON format
    The response body consists of Google cloud storage path of the images generated by LSTM microservice
    '''    
    closed_at_response = requests.post(LSTM_API_URL,
                                       json=closed_at_body,
                                       headers={'content-type': 'application/json'})
    
    '''
    Create the final response that consists of:
        1. GitHub repository data obtained from GitHub API
        2. Google cloud image urls of created and closed issues obtained from LSTM microservice
    '''
    json_response = {
        "created": created_at_issues,
        "closed": closed_at_issues,
        "starCount": repository["stargazers_count"],
        "forkCount": repository["forks_count"],
        "createdAtImageUrls": {
            **created_at_response.json(),
        },
        "closedAtImageUrls": {
            **closed_at_response.json(),
        },
    }
    # Return the response back to client (React app)
    return jsonify(json_response)


# Run flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
