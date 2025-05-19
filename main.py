# Written By: Michael Figueiredo
# Date:       2025-05-16
# Email:      MichaelFigueiredo743@gmail.com
# Description: This script is a machine learning pipeline that predicts vehicle emissions based on various features.
#  It includes data preprocessing, model training, and evaluation.


# imports
import pandas as pd
import numpy as np
# import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV #flashcards
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # Allows us to link the different pipelines together 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # For imputing missing values
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint

# Loading Data 
data = pd.read_csv('vehicle_emissions.csv')
data.head()
data.info()

# Create features and target variable               # we omit C02 emissions because were using x as a flash card. so 
x = data.drop(columns=['CO2_Emissions'], axis=1)  # create features, all the data except CO2 emissions, or target variable
y = data['CO2_Emissions']   # The asnwer key, the back of the flashcard

# Split Categorial and Numerical Features
numerical_cols = ["Model_Year", "Engine_Size", "Cylinders", "Fuel_Consumption_in_City(L/100 km)", "Fuel_Consumption_in_City_Hwy(L/100 km)", "Fuel_Consumption_comb(L/100km)", "Smog_Level"]
# Two lists, all the columns that are numerical and all the columns that are categorical
categorical_cols = ["Make", "Model", "Vehicle_Class", "Transmission",]


class ML_pipeline:
    def __init__(self, num_data, cat_data):
        self.num_data = num_data
        self.cat_data = cat_data
        self.best_params = None
        self.best_score = None
        self.cv_results = None
    
    def build_pipeline(self):

        # preprocessing numerical data
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),  # Impute missing values with mean
            ('scaler', StandardScaler())  # Scale the data, gibing it a mean and a variance
            ])
        # preprocessing categorical data
        categorial_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent ( since we cant take the average)
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables ( take strings and convert to numbers for model to understand)
            ])

        # Join the pipelines together
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols), # Here i have two pieces of the pipe and and cleaning the data
            ('cat', categorial_pipeline, categorical_cols)
            ])
        # Create the pipeline (final pipeline)
        pipeline = Pipeline([          
            ('preprocessor', preprocessor),  # Preprocess the data
            ('model', RandomForestClassifier())  # Model
        ])

        return pipeline

    
    def tuning(self, X_train, y_train, n_iter=100):
        param_distributions = {
            "model__n_estimators": randint(100, 500), # Number of trees in the forest
            "model__max_depth": [None] + list(range(10, 50, 10)),  # Maximum depth of the tree
            "model__min_samples_split": randint(2, 20),  # Minimum number of samples required to split an internal node
            "model__min_samples_leaf": randint(1, 10),  # Minimum number of samples required to be at a leaf node
            "model__max_features": ['auto', 'sqrt'],  # Number of features to consider when looking for the best split, whats gonna run through gridSearch. not everything in the world! (like gridSearch) but a random sample
            }

        pipeline = self.build_pipeline()  # Build the pipeline

        random_search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_distributions,
            n_iter=n_iter,  # Number of iterations
            cv=5,  # Number of cross-validation folds
            scoring='neg_mean_squared_error',  # Scoring metric
            n_jobs=-1,  # Use all available cores
            verbose=1,  # Verbosity level
            random_state=42  # Random seed for reproducibility
            )

        print(f'Randomixed Searh has started...')
        random_search.fit(X_train, y_train)

        # Save the results
        self.best_params = random_search.best_params_  # Best parameters
        self.best_score = random_search.best_score_  # Best score
        self.cv_results = random_search.cv_results_  # Cross-validation results

        return random_search.best_estimator_  # Return the best estimator
    


    def make_plots(self, model):
        feature_names = (numerical_cols + model.named_steps['preprocessor'].named_transformers_["cat"].named_steps
        ["encoder"].get_feature_names_out(categorical_cols).tolist())

        importance = model.named_steps['model'].feature_importances_  # Feature importance

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
            }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.show()



def main():

        
    testing = ML_pipeline(numerical_cols, categorical_cols)
       

     # Split into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)    # think of these as flashcards, the x is the front of flashcard
                                                                                            # y is the back of the flashcard
    best_model = testing.tuning(X_train, y_train, n_iter=100) 
            
        # Tuning the model with RandomizedSearchCV  
        # Fit the pipeline with the training data                 # WHen the model is learning, the model is allowed to view the front and back of flashcards. once we've trained the model and we 
        #pipeline.fit(X_train, y_train)                                                        # want to test what it's learned, we only show it the front of the flash card

    predictions = best_model.predict(X_test)                                                       #The model cannot see the back, it needs to see the front to make a prediction.


    mse = mean_squared_error(y_test, predictions)  # give it the back of the flashcard and the prediction      # meansqueared difference between the actual and predicted values
    rmse = np.sqrt(mse)  # root mean squared error
    mae = mean_absolute_error(y_test, predictions)  # absolute difference between the actual and predicted values
                                                 # the proportion of variance explained by the model
    r2 = r2_score(y_test, predictions)  # how well the model fits the data
       
    print(f'Model Performance Metrics:')
    print(f'R2 Score:{r2:.4f}') # 1.0 means perfect prediction
    print(f'Root Mean Square Error: {rmse:.2f}')  # lower is better
    print(f'Mean Absolute Error: {mae:.2f}')  # lower is better

    testing.make_plots(best_model)  # Plot the feature importance


if __name__ == "__main__":
    main()