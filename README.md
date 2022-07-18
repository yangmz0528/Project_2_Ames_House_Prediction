# Project 2 - Ames Housing Prediction

### Overview

`Primary Learning Objective`
*Creating and iterative refining a regression model
*Using Kaggle to practice the modeling process
*Providing business insights through reporting and presentation

### Problem Statement

We are a team of data scientists engaged by a real estate company to create a Machine Learning model that predicts the sale price of residential properties in Ames, Iowa with higher accuracy and more features as compared to available apps in the market and deploy it on an Application Programming Interface (API) for their agents. 

### Datasets

Listed below are the datasets included in the [`datasets`](./datasets/) folder for this project. 

Our model will be based on the Ames Housing Dataset ([link to Kaggle Dataset](https://www.kaggle.com/competitions/dsi-us-11-project-2-regression-challenge/data)) with estimated over 80 columns of different features relating to houses in Ames, Iowa from 2006 to 2010. The information of the 80 columns can be found [here](https://s3.amazonaws.com/dq-content/307/data_description.txt).

ome background story about Ames, Iowa:

Based on a United States Census Bureau report in 2010, Ames, Iowa has a population of approximately 59,000 and their economy is largely defined by Iowa State University. It is a public research university which is located in the middle of the city and most of the Ames's population are either students or faculty member of the Iowa State University.


### Exploratory Data Analysis

In this project, we will perform data cleaning procedure and visualise the general trend of the Sale price with many other features provided in the dataset.

### Regression Modelling

Linear Regression, Ridge Regression, Lasso Regression and ElasticNet modeling are performed to predict the house price and we will choose the best model to deploy to API. 

### Conclusion and Recommendation

In conclusion, the ridge model selected is able to produce a decent cross-validation score of 81% as compared to other regression models such as Linear, Lasso and ElasticNet and there is no issue of overfitting. This could be a relatively good model to predict the house price in Iowa. However, as the data has been removed for data with above 400k sale price, this model could only make relatively good predictions of the house in Ames below 400k. 

There are also other information that the dataset could include such as crime rate within the neighborhood, latitiude/longitude of the neighborhood location so that we can visualise better of the housing distribution in Ames, Iowa.

Beside all these, the model generally is able to make good prediction of the house price and value add on to current house listng application in the market by having more features for users to select. Once the model is successfully deployed, property agents can use our model to forsee undervalued houses and see how different features can maximise their profits through renovations etc. In additional, for citizens who just wanted to buy a house can also use our model to see whether the house is overpriced or not overpriced.  

Last but not least, for our future project considerations, we might want to consider the feature importance of the model through coefficients so as to see which are the features that are impacting our model the most. Property surveyor could also be engaged for evaluating the price of houses such that our model and their evaluation can compliments each other.