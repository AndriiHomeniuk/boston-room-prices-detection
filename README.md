# Boston Room Prices

I used Apache Spark’s spark.ml Linear Regression for predicting Boston housing prices. 
Our data is from the Kaggle competition: Housing Values in Suburbs of Boston. 
For each house observation, we have the following information:
- CRIM — per capita crime rate by town.
- ZN — proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS — proportion of non-retail business acres per town.
- CHAS — Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
- NOX — nitrogen oxides concentration (parts per 10 million).
- RM — average number of rooms per dwelling.
- AGE — proportion of owner-occupied units built prior to 1940.
- DIS — weighted mean of distances to five Boston employment centres.
- RAD — index of accessibility to radial highways.
- TAX — full-value property-tax rate per $10,000.
- PTRATIO — pupil-teacher ratio by town.
- BLACK — 1000(Bk — 0.63)² where Bk is the proportion of blacks by town.
- LSTAT — lower status of the population (percent).
- MEDV — median value of owner-occupied homes in $1000s. This is the target variable.

The input data set contains data about details of various houses. Based on the information provided, the goal is to come up with a model to predict median value of a given house in the area.