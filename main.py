from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    GBTRegressor,
)
from pyspark.sql import SQLContext

from utils import update_columns


def preparation_data(df_columns, main_df):
    main_df = update_columns(df_columns, main_df)
    vector_assembler = VectorAssembler(
        inputCols=['CRIM_Vec', 'ZN_Vec', 'INDUS_Vec', 'CHAS_Vec', 'NOX', 'RM', 'AGE_Vec', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT_Vec'],
        outputCol='features',
    )
    vhouse_df = vector_assembler.transform(main_df).select(['features', 'MEDV'])
    return vhouse_df.randomSplit([0.7, 0.3])


def linear_regression(train_data, test_data):
    lr = LinearRegression(
        featuresCol='features',
        labelCol='MEDV',
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
    )
    lr_model = lr.fit(train_data)
    print(f'Coefficients: {str(lr_model.coefficients)}')
    print(f'Intercept: {str(lr_model.intercept)}')
    training_summary = lr_model.summary

    #  RMSE measures the differences between predicted values by the model and the actual values. However, RMSE alone is meaningless until we compare with the actual “MEDV” value, such as mean, min and max. After such comparison, our RMSE looks pretty good.
    print(f'RMSE: {training_summary.rootMeanSquaredError}')
    #  R squared at 0.62 indicates that in our model, approximate 74% of the variability in “MEDV” can be explained using the model.
    print(f'r2: {training_summary.r2}')
    print(train_data.describe().show())

    lr_predictions = lr_model.transform(test_data)
    print(lr_predictions.select('prediction', 'MEDV', 'features').show(5))
    lr_evaluator = RegressionEvaluator(
        predictionCol='prediction',
        labelCol='MEDV',
        metricName='r2',
    )
    print('R Squared (R2) on test data = %g' % lr_evaluator.evaluate(lr_predictions))

    test_result = lr_model.evaluate(test_data)
    print('Root Mean Squared Error (RMSE) on test data = %g' % test_result.rootMeanSquaredError)
    print('numIterations: %d' % training_summary.totalIterations)
    print(f'objectiveHistory: {str(training_summary.objectiveHistory)}')
    print(training_summary.residuals.show(5))

    predictions = lr_model.transform(test_data)
    print(predictions.select('prediction', 'MEDV', 'features').show(5))


def decision_tree_regression(train_data, test_data):
    dt = DecisionTreeRegressor(featuresCol='features', labelCol='MEDV')
    dt_model = dt.fit(train_data)
    dt_predictions = dt_model.transform(test_data)
    dt_evaluator = RegressionEvaluator(
        labelCol='MEDV',
        predictionCol='prediction',
        metricName='rmse',
    )
    rmse = dt_evaluator.evaluate(dt_predictions)
    print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)
    print(dt_model.featureImportances)
    # Apparently, the number of rooms is the most important feature to predict the house median price in our data.


def gradient_boosted_tree_regression(train_data, test_data):
    gbt = GBTRegressor(featuresCol='features', labelCol='MEDV', maxIter=10)
    gbt_model = gbt.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)
    print(gbt_predictions.select('prediction', 'MEDV', 'features').show(5))
    gbt_evaluator = RegressionEvaluator(
        labelCol='MEDV',
        predictionCol='prediction',
        metricName='rmse',
    )
    rmse = gbt_evaluator.evaluate(gbt_predictions)
    print('Root Mean Squared Error (RMSE) on test data = %g' % rmse)


if __name__ == '__main__':
    sqlContext = SQLContext(SparkContext())
    house_df = sqlContext.read \
        .format('com.databricks.spark.csv') \
        .options(header='true', inferschema='true') \
        .load('boston_room_prices/boston.csv')
    house_df.cache()

    schema = 'CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV'
    string_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']
    train_df, test_df, *_ = preparation_data(string_columns, house_df)

    linear_regression(train_df, test_df)
    decision_tree_regression(train_df, test_df)
    gradient_boosted_tree_regression(train_df, test_df)
