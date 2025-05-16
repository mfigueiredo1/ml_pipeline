# Written By: Michael Figueiredo
# Date:       2025-05-16
# Email:      MichaelFigueiredo743@gmail.com
# Description: This script is a machine learning pipeline that predicts vehicle emissions based on various features.
#  It includes data preprocessing, model training, and evaluation.


# imports
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split  #flashcards
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # Allows us to link the different pipelines together 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # For imputing missing values
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Loading Data 
data = pd.read_csv('vehicle_emissions.csv')
data.head()
data.info()

# Create features and target variable               # we omit C02 emissions because were using x as a flash card. so 
x = data.drop(columns=['CO2_Emissions'], axis=1)
y = data['CO2_Emissions']

# Split Categorial and Numerical Features
numerical_cols = ["Model_Year", "Engine_Size", "Cylinders", "Fuel_Consumption_in_City(L/100 km)", "Fuel_Consumption_in_City_Hwy(L/100 km)", "Fuel_Consumption_comb(L/100km)", "Smog_Level"]

categorical_cols = ["Make", "Model", "Vehicle_Class", "Transmission",]


# Start the Pipeline w/ Enoding
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Scale the data
])

categorial_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables
])

# Join the pipelines together
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorial_pipeline, categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocess the data
    ('model', RandomForestClassifier())  # Model
])

# Split into training and testing datasets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)    # think of these as flashcards, the x is the front of flashcard
                                                                                            # y is the back of the flashcard

# Train and predict Model                               # WHen the model is learning, the model is allowed to view the front and back of flashcards. once we've trained the model and we 
pipeline.fit(X_train, y_train)                                                        # want to test what it's learned, we only show it the front of the flash card

prediction = pipeline.predict(X_test)                                                       #The model cannot see the back, it needs to see the front to make a prediction.

#View the encoding thats done                                                                     # then when we go through to evaluate the model, we can see how well it did by comparing the predictions to the actual values
encoded_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
print(encoded_cols)

# Evaluate model accuracy

mse = mean_squared_error(y_test, prediction)  # give it the back of the flashcard and the prediction      # meansqueared difference between the actual and predicted values
rmse = np.sqrt(mse)  # root mean squared error

r2 = r2_score(y_test, prediction)  # how well the model fits the data
mae = mean_absolute_error(y_test, prediction)  # absolute difference between the actual and predicted values
                                                 # the proportion of variance explained by the model

print(f'Model Performance:')
print(f'R2 Score:{r2}')
print(f'Root Mean Square Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

joblib.dump(pipeline, "vehicle_emissions_pipeline.joblib")  # Save the model
