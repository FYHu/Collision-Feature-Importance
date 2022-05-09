# Package Guide
Please install the following packages before runing the code.

- Python3
- tensorflow
- sklearn
- numpy
- matplotlib
- imblearn
- keras
- seaborn
- pandas


# Source Code Description


## Data Structure
```bash
|
|-- data
|    |-- 2015
|        |-- 2015-gla-data-extract-attendant.csv
|        |-- 2015-gla-data-extract-casualty.csv
|        |-- 2015-gla-data-extract-vehicle.csv
|    |-- 2016
|    |-- 2017
|    |-- 2018    
|    |-- 2019  
|    |-- 2020  
|
|-- src
    | ...


```
## Data Link

- TFL Collision Data: https://tfl.gov.uk/corporate/publications-and-reports/road-safety
- Weather Historical Data: https://www.timeanddate.com/weather/uk/london/historic

## File Description

- `construct_random_subsample`: Used to construct random subsample of data, to deal with data imblance
- `exceptions`: Excpetions
- `feature_importance`: Plot Feature importance(Top10) Helper Function
- `load_data`: Load the entire dataframe and preprocess the data
- `model_evaluation`: Model evaluation helper function
- `random_forest`: Run Random Forest Model(best model) and evaluate[Accurary, F1 Score, Confusion Matrix, Random Permutation Feature Importance]
- `rf_competition`: Random forest Competition, find the best random forest model to deal with imbalance data
- `time_series_gridSearchCV`: Self Modified gridsearch cross validation with time series data and time series split, reuturn the best parameter with the highest f1 scroe
- `help_fucntion`: Helper Functions
- `train_test_split`: Train Test Split Helper Function
- `feed_forward_nn`: Run Feed Forward Neural Network Model(best hyperparameter) and evaluate[Accurary, F1 Score, Confusion Matrix, Random Permutation Feature Importance]
- `weighted_lr`: Run weighted Logistic Regression(best hyperparameter) and evaluate[Accurary, F1 Score, Confusion Matrix, coefficient feature importance]
- `cross_entropy`: cross entropy loss function, build for evaluating both training and testing loss



