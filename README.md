# flight-ticket-price-estimation.
Flight Ticket Price Estimation is a machine learning project designed to estimate the price of a flight ticket based on factors such as airline, source, destination, travel month, number of days before booking, number of stops, and flight duration. The project uses Python, Pandas, Scikit-learn, and Matplotlib.

## 1. Project Overview

Flight Ticket Price Estimation is a Machine Learning project that predicts the estimated price of a flight ticket based on different flight-related factors.

The project uses Python and the Scikit-learn library to build a Linear Regression model. It takes information such as airline, source, destination, travel month, booking time, number of stops, and flight duration as input and predicts the expected ticket price.

## 2. Objective

The main objective of this project is to develop a machine learning model that can estimate flight ticket prices from historical flight data.

The project helps understand how different factors can influence the price of airline tickets.

## 3. Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Linear Regression
* Data Preprocessing
* Machine Learning

## 4. Dataset

The program reads the dataset from:

`flight_data.csv`

The target variable is:

`ticket_price`

The input features used by the model are:

* Airline
* Source
* Destination
* Travel Month
* Days Before Booking
* Stops
* Duration in Hours

The `flight_id` column is removed because it is an identifier and is not required for price prediction.

## 5. Machine Learning Algorithm

The project uses:

**Linear Regression**

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

In this project, the continuous value being predicted is the flight ticket price.

## 6. Data Preprocessing

The project uses Scikit-learn preprocessing techniques.

### Numerical Features

The numerical features are:

* `travel_month`
* `days_before_booking`
* `duration_hours`

Missing numerical values are replaced using the median value.

The numerical features are then standardized using `StandardScaler`.

### Categorical Features

The categorical features are:

* `airline`
* `source`
* `destination`
* `stops`

Missing categorical values are replaced using the most frequently occurring value.

The categorical features are converted into numerical form using `OneHotEncoder`.

## 7. Model Pipeline

A Scikit-learn Pipeline is used to combine preprocessing and Linear Regression into one workflow.

The pipeline contains:

1. Data preprocessing
2. Numerical feature processing
3. Categorical feature processing
4. Linear Regression model

This makes the machine learning workflow easier to manage.

## 8. Train-Test Split

The dataset is divided into two parts:

* 80% Training Data
* 20% Testing Data

The training data is used to train the model, while the testing data is used to evaluate its performance.

## 9. Model Evaluation

The project uses three evaluation metrics:

### MAE

Mean Absolute Error measures the average difference between actual and predicted ticket prices.

### RMSE

Root Mean Squared Error measures prediction error while giving more importance to larger errors.

### R² Score

R² Score indicates how well the model explains the variation in ticket prices.

The program displays all three values after testing the model.

## 10. Actual vs Predicted Prices

The project creates a comparison between:

* Actual Ticket Price
* Predicted Ticket Price

This allows us to see how close the model's predictions are to the actual prices.

## 11. New Flight Prediction

The trained model can be used to estimate the price of a new flight.

The example in the program uses:

* Airline: Vistara
* Source: Delhi
* Destination: Mumbai
* Travel Month: December
* Days Before Booking: 30
* Stops: Non-stop
* Duration: 2.2 hours

The model predicts the estimated ticket price for this flight.

## 12. Feature Impact Analysis

The project calculates the coefficients of the Linear Regression model.

The absolute value of each coefficient is used to identify features that have a larger impact on the prediction.

The program displays the top 15 factors according to their absolute impact.

## 13. Data Visualization

The project creates several graphs.

### Flight Duration vs Ticket Price

Shows the relationship between flight duration and ticket price.

### Booking Days vs Ticket Price

Shows the relationship between the number of days before booking and ticket price.

### Average Ticket Price by Airline

Shows the average ticket price for different airlines.

### Actual vs Predicted Prices

Compares actual ticket prices with the prices predicted by the machine learning model.

## 14. Project Workflow

Dataset
↓
Data Loading
↓
Data Inspection
↓
Feature Selection
↓
Data Preprocessing
↓
Train-Test Split
↓
Linear Regression
↓
Model Training
↓
Price Prediction
↓
Model Evaluation
↓
Visualization

## 15. Main Python Libraries

### Pandas

Used for loading and handling the flight dataset.

### Matplotlib

Used to create graphs and visualize relationships in the data.

### Scikit-learn

Used for preprocessing, splitting data, building the machine learning model, and evaluating predictions.

## 16. Advantages

* Simple and easy-to-understand machine learning project
* Uses both numerical and categorical features
* Handles missing values
* Converts categorical data into numerical form
* Provides multiple model evaluation metrics
* Can predict prices for new flights
* Provides useful visualizations
* Helps identify important factors affecting ticket prices

## 17. Limitations

* Linear Regression assumes a linear relationship between features and ticket price.
* Ticket prices can change because of many external factors that are not included in the dataset.
* The prediction quality depends on the quality and size of the dataset.
* The current project does not compare multiple machine learning algorithms.

## 18. Future Enhancements

The project can be improved by:

* Using larger and more recent flight datasets
* Comparing Linear Regression with Random Forest, Decision Tree, and other algorithms
* Adding more flight-related features
* Creating a graphical user interface
* Developing a web application for price prediction
* Adding real-time flight data
* Improving model accuracy through hyperparameter tuning

## 19. Expected Output

The program displays:

* Dataset information
* Dataset shape
* Missing values
* MAE
* RMSE
* R² Score
* Actual and predicted ticket prices
* Estimated price for a new flight
* Top factors affecting ticket price

It also generates four visualization files:

* `duration_vs_price.png`
* `booking_days_vs_price.png`
* `airline_vs_price.png`
* `actual_vs_predicted.png`

## 20. Conclusion

The Flight Ticket Price Estimation project demonstrates how Machine Learning can be used to predict flight ticket prices. It uses data preprocessing, categorical encoding, numerical scaling, Linear Regression, model evaluation, feature analysis, and visualization.

This project is suitable as a beginner-level Machine Learning project because it demonstrates the complete workflow from loading a dataset to making predictions and evaluating the model.
