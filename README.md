# INST414project
Project: Predictive Energy Forecasting
Summary: Using machine learning techniques such as Random Forest model, the following project works to predict energy consumption and pricing trends in Maryland. By analyzing historical data, this project aims to enhance grid management, reduce operational inefficiencies, and support more sustainable energy practices. With accurate predictions, utility providers can optimize energy distribution, minimize waste, and reduce the environmental impact of overproduction.
This project focuses on forecasting energy consumption in Maryland to enhance grid management and promote energy efficiency. By leveraging advanced data analytics and machine learning techniques, the goal is to predict energy demand with greater accuracy. In a world where energy consumption fluctuates due to various factors such as weather, time of day, and economic activity, accurate forecasting is critical to maintaining a stable and reliable energy grid.
The primary problem this project addresses is the inability of utility providers to predict energy consumption accurately, which can lead to power shortages, grid instability, and unnecessary overproduction of energy. These disorganizations result in higher operational costs, environmental harm, and strain on energy infrastructure. By creating a predictive model, this project aims to provide a solution that enables utility providers to proactively adjust their energy production, optimize distribution, and contribute to a more sustainable energy system.
Features of the project will include the following:
Energy Consumption Forecasting: The project predicts future energy consumption in Maryland based on historical data, helping utility providers anticipate demand fluctuations.
Energy Price Prediction: The model forecasts energy prices for the following year, assisting in planning and cost optimization for utility providers.
Machine Learning Model: Utilizes the Random Forest algorithm to predict energy consumption and prices, ensuring accurate results through cross-validation.
Data Normalization: Data used in the project is normalized before model training to improve the accuracy and performance of the predictions.
Performance Evaluation: Uses Mean Absolute Error (MAE) to measure the accuracy of the predictions and assess model performance.
Data Visualization: Provides visual comparisons between predicted and actual data using bar graphs, making it easier to understand and evaluate the model’s predictions. Visualizations include mainly bar charts to help compare two factors.
Sustainability Focus: Aims to reduce reliance on fossil fuels, minimize carbon emissions, and contribute to more sustainable energy distribution practices.
State-Specific Insights: Focuses on energy consumption patterns in Maryland, providing valuable local insights for better grid management and planning.
For this project several Python libraries will be usedfor data manipulation, machine learning model development, and evaluation. The following is the list of libraries required for this project:
Pandas: For data manipulation and analysis.
import pandas as pd
NumPy: For numerical computations and handling arrays.
import numpy as np 
Matplotlib: For visualizing data and plotting graphs.
import matplotlib.pyplot as plt
Scikit-learn: For machine learning, model evaluation, and preprocessing.
from sklearn.ensemble import RandomForestRegressor  →  Random Forest algorithm for regression modeling
from sklearn.preprocessing import StandardScaler →  For scaling data
from sklearn.model_selection import cross_val_score →  For cross-validation
from sklearn.model_selection import KFold   →  For K-Fold cross-validation
Usage:
In order to interact with this project, users can run the Python scripts provided for data analysis and model training. After setting up the environment and installing the necessary libraries, you can start by running the main script, which loads the dataset, preprocesses the data, trains the Random Forest model, and generates visualizations to compare predicted and actual energy consumption. Below is an example command to run the script:
python energy_forecasting.py
Or through the jupyter Lab:
python energy_forecasting.ipynb by clicking the run button for each cell of code.
Configuration:
There are no complex configuration files required for this project. However, if the user wishes to customize the data processing or model parameters, you can modify the variables within the script including the data that the user would like to experiment with as well as the model parameter. The parameters of the Random Forest model can be changed based on the dataset and desired results which include features such as maximum depth.
Data:
This project uses energy consumption data for residential areas in Maryland, sourced from the U.S. Energy Information Administration (EIA). The dataset contains historical data on energy consumption patterns, which is essential for forecasting future consumption and energy prices. To obtain the dataset, you can download it directly from the EIA website(https://www.eia.gov/consumption/residential/data/2020/index.php?view=microdata). 
This project uses a manually input dataset based on energy consumption data for the years 2018 to 2023. The dataset contains the year, Revenue (thousand dollars), Avg Price (cents/kWh), and Total retail sales (MWh). Users can modify the dataset by updating the values in the dictionary for further analysis or use a different dataset by replacing the values with their own data.
Model & Analysis:
For this project, the Random Forest model was used to predict the total retail sales of electricity (in megawatt-hours) for the years 2021, 2022, and 2023, based on the data from previous years (2018–2020). The methodology for applying this model involved the following steps:
Data Preparation: The dataset includes several key features, including the year, average price of electricity in cents per kilowatt-hour (kWh), and the actual and predicted retail sales. For the model, the "Avg Price (cents/kWh)" and "Total retail sales (MWh)" were the key input variables, while the "Predicted retail sales (MWh)" was the target variable for the model to predict.
Model Selection: The Random Forest model was chosen due to its ability to handle complex, non-linear relationships in the data. It is an ensemble method that combines the predictions of multiple decision trees to improve accuracy and reduce overfitting. Random Forest is especially effective in handling large datasets with various features and can also provide insights into feature importance.
Training the Model: The model was trained using historical data (2018–2020) to learn the patterns and relationships between the input features (avg price and total retail sales) and the target variable (predicted retail sales). The Random Forest algorithm was tuned with default parameters, but hyperparameter tuning could be performed to improve accuracy further.
Prediction: After training, the model predicts the retail sales (in MWh) for the years 2021, 2022, and 2023, based on the patterns learned from the training data. These predictions will be visualized alongside the actual values from the data source for comparison.
Cross-validation: To assess the model’s performance, cross-validation was conducted to evaluate how well the model generalizes to unseen data. This technique splits the data into multiple subsets (folds), trains the model on some folds, and validates it on the remaining folds. It helps ensure that the model is not overfitting to a specific subset of the data.
Mean Absolute Error (MAE): MAE was calculated to measure the accuracy of the model's predictions. MAE provides the average difference between the predicted and actual values, allowing for a clearer understanding of the model’s prediction errors. A lower MAE indicates a more accurate model.
Testing:
Testing in the project was performed to validate the accuracy of the Random Forest model's predictions for the years 2021, 2022, and 2023. The following steps were used to test the model:
Test Setup: The model was tested on the data for the years 2021, 2022, and 2023, using all available historical data from previous years (2018–2020) for training.
Cross-validation: Cross-validation was implemented to evaluate the model’s performance on different subsets of the data. This ensures that the model’s predictions are not biased by any specific subset of the data.
MAE Calculation: The mean absolute error (MAE) was computed to quantify the average error between the predicted and actual retail sales for the test years. A lower MAE value indicates that the model's predictions are more accurate.
Testing Framework: The testing was performed using python's scikit-learn library, which provides the necessary tools for training, validating, and testing machine learning models. Specific functions used include RandomForestRegressor for the model, cross_val_score for cross-validation, and mean_absolute_error for calculating MAE.
Visualizing the predicted retail sales against the actual retail sales for the test years, 2021-2023, this model is able to be evaluated for accuracy. Thus, discrepancies between predicted values and actual values can further be analyzed to improve any models that are done in the future.
Contributors:
Rasanjali Ilandara Devage: Primary Contributor, developed the project, including data collection, model selection, analysis, and testing. Responsible for designing the predictive model using Random Forest and performing cross-validation and MAE analysis
License:
Python-This project  used Python, licenced under Python Software Foundation License
Visual Studio-This project was developed using Microsoft Visual Studio 


