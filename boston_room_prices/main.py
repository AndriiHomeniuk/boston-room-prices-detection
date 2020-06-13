from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.sql import SQLContext

from boston_room_prices.utils import update_columns

sc = SparkContext()
sqlContext = SQLContext(sc)
house_df = sqlContext.read\
    .format('com.databricks.spark.csv')\
    .options(header='true', inferschema='true')\
    .load('boston_room_prices/boston.csv')
house_df.cache()

schema = 'CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV'
string_columns = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'AGE',
    'LSTAT',
]

# Prepare raw data

house_df = update_columns(string_columns, house_df)
vectorAssembler = VectorAssembler(
    inputCols=['CRIM_Vec', 'ZN_Vec', 'INDUS_Vec', 'CHAS_Vec', 'NOX', 'RM', 'AGE_Vec', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT_Vec'],
    outputCol='features',
)
vhouse_df = vectorAssembler.transform(house_df).select(['features', 'MEDV'])
train_df, test_df, *_ = vhouse_df.randomSplit([0.7, 0.3])

# Linear Regression

lr = LinearRegression(
    featuresCol='features', 
    labelCol='MEDV', 
    maxIter=10, 
    regParam=0.3, 
    elasticNetParam=0.8,
)
lr_model = lr.fit(train_df)
print('Coefficients: ' + str(lr_model.coefficients))
print('Intercept: ' + str(lr_model.intercept))
trainingSummary = lr_model.summary

#  RMSE measures the differences between predicted values by the model and the actual values. However, RMSE alone is meaningless until we compare with the actual “MEDV” value, such as mean, min and max. After such comparison, our RMSE looks pretty good.
print('RMSE: %f' % trainingSummary.rootMeanSquaredError)
#  R squared at 0.62 indicates that in our model, approximate 74% of the variability in “MEDV” can be explained using the model.
print('r2: %f' % trainingSummary.r2)
print(train_df.describe().show())

lr_predictions = lr_model.transform(test_df)
print(lr_predictions.select('prediction', 'MEDV', 'features').show(5))
lr_evaluator = RegressionEvaluator(
    predictionCol='prediction',
    labelCol='MEDV',
    metricName='r2',
)
print('R Squared (R2) on test data = %g' % lr_evaluator.evaluate(lr_predictions))
test_result = lr_model.evaluate(test_df)
print('Root Mean Squared Error (RMSE) on test data = %g' % test_result.rootMeanSquaredError)
print('numIterations: %d' % trainingSummary.totalIterations)
print('objectiveHistory: %s' % str(trainingSummary.objectiveHistory))
print(trainingSummary.residuals.show())

predictions = lr_model.transform(test_df)
print(predictions.select('prediction', 'MEDV', 'features').show())

# Decision tree regression

dt = DecisionTreeRegressor(featuresCol='features', labelCol='MEDV')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(
    labelCol='MEDV',
    predictionCol='prediction',
    metricName='rmse',
)
rmse = dt_evaluator.evaluate(dt_predictions)
print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)
print(dt_model.featureImportances)
# Apparently, the number of rooms is the most important feature to predict the house median price in our data.

# Gradient-boosted tree regression

gbt = GBTRegressor(featuresCol='features', labelCol='MEDV', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
print(gbt_predictions.select('prediction', 'MEDV', 'features').show(5))
gbt_evaluator = RegressionEvaluator(
    labelCol='MEDV',
    predictionCol='prediction',
    metricName='rmse',
)
rmse = gbt_evaluator.evaluate(gbt_predictions)
print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)
