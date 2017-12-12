# Dengue Predictor (Team 25)

A project using machine learning models to predict the dengue outbreak and give Visualization of the predition geographically using google maps.

# Project Design

#### Machine Learning Model Training Phase

Its a machine learning model which will be considering multiple factors such as rainfall, temprature and other weather related data.We used Numpy and Pandas to clean the data and bring it in structured format. Specially we used the average (minfill) technique to make data more cleaner. After cleaning the data and visualizing the ploted charts from it, we decided tp go ahed with linear regression algorythm to predict the data.
For Development phase we have used 70% of raw data to train our Machine learning model.

#### Machine Learning Model Testing Phase

For Testing, we have used the rest 30% data and applied our Prediction model on it. The prediction result, after comparing with actual data, was looking quite promissing. We have successfully tested the model and then applied that to a new weather related data and predicted the output.

#### Successful Prediction

After Successful testing and promising results, we went ahead and applied the weather realated data to our model and produced the prediction output. The output which we are getting after prediction is in CSV format defining prediction for per year, per week.

#### Visualization Phase using Google Maps API
This is the Visualizer API which allows users to get the visualization of the Dengue outbreak geographically. 

##### It contains 2 parts. 
* Prediction Data Upload
* Visualizaation of data on Google Maps 

Users are allowed to upload the prediction data csv (it can be ours or can be from other prediction). The web module will fetch the data from CSV and put it into the MySql database according to the required format. The prediction data along with the citi's longitude and latitude data is used to show the Visualization on the google map. It contains all the functionalities which google map is giving along with our custom facilities to narrow down the search starting from year to week of year.


# Libraries Used

#### Data cleaning 
Pandas, numpy.

#### Data Ploting
matplotlib, seaborn charts

#### Visualization
Googlr Maps API


# Prediction Algorythm
linear regression algorithms
