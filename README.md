# Populareddit

This tool was created by Davide Martizzi as a project for the Insight Data Science Fellowship, Silicon Valley 2019C.

The tool predicts whether a given text post will be popular in a given subreddit. 

The predictive model is based on a combination of topic modeling with non-negative matrix factorization (NMF) and gradient boosting regression. 

# Requirements 

The code runs on Python 3.x. The required packages are listed in `requirements.txt`

# Jupyter Notebooks

-`populareddit-pull-subreddit-data.ipynb` uses PRAW and/or Pushshift.io and/or the Reddit API to pull submissions in a given subreddit. 

-`populareddit-train-model.ipynb` trains topic models based on NMF and regression models based on gradient boosting for a given subreddit. 

-`populareddit-topic-visualization.ipynb` experiments with topic visualization with the `pyLDAvis` package. 

# Web App - Demo Version

Production models for a few subreddits were deployed on a web app. This version of the app should not be considered more than a demo.

Feel free to play with [PopulaReddit](http://populareddit.xyz/).

# Notes

The accuracy of the model predicting popularity is between 60% and 85%, depending on which subreddit is selected. Improvements on the way...
