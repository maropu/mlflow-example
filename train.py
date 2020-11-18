"""
A sample code of MLflow for nttlabs in Medium
"""

import os
import sys
import zipfile

import numpy as np
# import shap
import xgboost as xgb
from argparse import ArgumentParser
from pyspark.sql import SparkSession
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve

import mlflow
import mlflow.xgboost

print('MLflow version:%s' % mlflow.version.VERSION)
print('Tracking URI:%s' % mlflow.tracking.get_tracking_uri())

def check_if_dataset_exists():
    data_name = 'RealWorldDatasets'
    if not os.path.isdir('./dataset/%s' % data_name):
        print("Downloading '%s' and saving it in ./dataset" % data_name)
        url = 'http://cseweb.ucsd.edu/~arunkk/hamlet/%s.zip' % data_name
        savePath = './dataset/%s.zip' % data_name
        urlretrieve(url, savePath)
        with zipfile.ZipFile(savePath) as zip:
            zip.extractall(path='./dataset')

if __name__ == "__main__":
    # Parses command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', help='Expriment Name', type=str, required=False, default='Default')
    parser.add_argument('--run_name', dest='run_name', help='Run Name', type=str, required=False, default='N/A')
    parser.add_argument('--max_depth', dest='max_depth', help='XGBoost Max Depth', type=str, required=True)
    parser.add_argument('--learning_rate', dest='learning_rate', help='XGBoost Learning Rate', type=str, required=True)
    parser.add_argument('--subsample', dest='subsample', help='XGBoost Subsampling Ratio', type=str, required=True)
    args = parser.parse_args()

    # Checks if the dataset exists
    check_if_dataset_exists()

    # Set the name for this training
    print('experiment_name:%s' % args.experiment_name)
    mlflow.set_experiment(args.experiment_name)

    # Loads the Walmart data whose join graph is as follows;
    #                                                        _ R2_stores{store, ...}
    #                                                       /
    #  S_sales{weekly_sales, sid, dept, ..., purchaseid, store}
    #                                            /
    #           R1_indicators{purchaseid, ...} _/
    #
    spark = SparkSession.builder.appName('walmart').getOrCreate()
    base_path = './dataset/RealWorldDatasets/WalMart/'
    spark.read.option('header', True).option('inferSchema', True).csv(base_path + 'S_sales.csv').createOrReplaceTempView('S_sales')
    spark.read.option('header', True).option('inferSchema', True).csv(base_path + 'R1_indicators.csv').createOrReplaceTempView('R1_indicators')
    spark.read.option('header', True).option('inferSchema', True).csv(base_path + 'R2_stores.csv').createOrReplaceTempView('R2_stores')
    sdf = spark.sql(
        "SELECT " \
            "CAST(TRIM(BOTH '\\'' FROM weekly_sales) AS INT) weekly_sales, " \
            "CAST(TRIM(BOTH '\\'' FROM sid) AS INT) sid, " \
            "CAST(TRIM(BOTH '\\'' FROM dept) AS INT) dept, " \
            "CAST(TRIM(BOTH '\\'' FROM s.store) AS INT) store, " \
            "CAST(TRIM(BOTH '\\'' FROM type) AS INT) type, " \
            "size, " \
            "temperature_stdev, " \
            "fuel_price_avg, " \
            "fuel_price_stdev, " \
            "cpi_avg, " \
            "cpi_stdev, " \
            "unemployment_avg, " \
            "unemployment_stdev, " \
            "holidayfreq " \
        "FROM " \
            "S_sales s, " \
            "R1_indicators i, " \
            "R2_stores st " \
        "WHERE " \
            "s.purchaseid = i.purchaseid AND " \
            "s.store = st.store")

    # Converts into a Pandas DataFrame
    df = sdf.toPandas()

    spark.stop()

    # Splits `df` into two parts: training (`X_train` and `y_train`) and
    # test data (`X_test` and `y_test`)
    X = df[df.columns[df.columns != 'weekly_sales']]
    y = df['weekly_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    # Enables automatic logging for XGBoost
    mlflow.xgboost.autolog()

    # Starts runs with different XGBoost parameters
    for md in args.max_depth.split(','):
        for lr in args.learning_rate.split(','):
            for ssr in args.subsample.split(','):
                # Creates an execution context for a single run with given parameters (`md`, `lr`, and `ssr`)
                with mlflow.start_run(run_name=args.run_name) as run:
                    clf = xgb.XGBClassifier(max_depth=int(md), learning_rate=float(lr), nthread=-1, subsample=float(ssr))
                    clf.fit(X_train, y_train)

                    # Computes a metric for the built model
                    pred = clf.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))

                    # For better tracking, stores the training logs and the built model
                    # in the MLflow logging framework
                    # TODO: Saves a graphviz image for feature importances in XGBoost
                    from mlflow.models.signature import infer_signature
                    infer_signature(X_train, y_test)

                    # This feature implemented in MLflow v1.12.0
                    # mlflow.shap.log_explanation(clf, X_train)

                    mlflow.set_tag('training algorithm', 'xgboost')
                    mlflow.log_metrics({'RMSE': rmse})
                    mlflow.xgboost.log_model(clf, 'model')

                    print('XGBoost model (max_depth=%s, learning_rate=%s, subsample=%s):' % (md, lr, ssr))
                    print('  RMSE: %f' % rmse)

