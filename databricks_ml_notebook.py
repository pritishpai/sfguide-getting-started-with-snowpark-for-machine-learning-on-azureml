# Databricks notebook source
# MAGIC %md
# MAGIC # Working in Databricks

# COMMAND ----------

# create spark dataframes from CSV files

import pandas as pd
import logging
import pyspark.sql.functions as F

# Load the CSV files required for ML
maintenance_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("file:/Workspace/Repos/pritish.pai@databricks.com/sfguide-getting-started-with-snowpark-for-machine-learning-on-azureml/maintenance.csv")
humidity_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("file:/Workspace/Repos/pritish.pai@databricks.com/sfguide-getting-started-with-snowpark-for-machine-learning-on-azureml/humidity.csv")
hum_udi_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("file:/Workspace/Repos/pritish.pai@databricks.com/sfguide-getting-started-with-snowpark-for-machine-learning-on-azureml/city_udi.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC # Look at the dataframes

# COMMAND ----------

maintenance_df.head()

# COMMAND ----------

humidity_df.head()

# COMMAND ----------

hum_udi_df.head()

# COMMAND ----------

# join together the dataframes and prepare training dataset
from pyspark.sql.functions import col

maintenance_city = maintenance_df.join(hum_udi_df, ["UDI"])
maintenance_hum = maintenance_city.join(humidity_df, (maintenance_city["CITY"] == humidity_df["CITY_NAME"])).select(
    col("TYPE"), 
    col("AIR_TEMPERATURE_K"), 
    col("PROCESS_TEMPERATURE"), 
    col("ROTATIONAL_SPEED_RPM"), 
    col("TORQUE_NM"), 
    col("TOOL_WEAR_MIN"), 
    col("HUMIDITY_RELATIVE_AVG"), 
    col("MACHINE_FAILURE"))

# COMMAND ----------

# create the required catalogs and databases
spark.sql("CREATE CATALOG IF NOT EXISTS ppai_poc_capstone")
spark.sql("USE CATALOG ppai_poc_capstone")
spark.sql("CREATE DATABASE IF NOT EXISTS machine_predictive_maintenance")
spark.sql("USE DATABASE machine_predictive_maintenance")


# using overwrite mode for re-runnability of POC
maintenance_hum.write.mode("overwrite").format('delta').saveAsTable("maintenance_hum")
maintenance_df.write.mode("overwrite").format('delta').saveAsTable("maintenance")
humidity_df.write.mode("overwrite").format("delta").saveAsTable("humidity")
hum_udi_df.write.mode("overwrite").format("delta").saveAsTable("city_udf")



# COMMAND ----------

# drop column thats not needed
maintenance_hum_df = maintenance_hum.drop("TYPE")

# COMMAND ----------

# MAGIC %md
# MAGIC # Use MLFlow to track jobs and models

# COMMAND ----------

import mlflow
from mlflow import log_metric, log_param, log_artifacts

# create a new experiment
exp_name = "/Users/pritish.pai@databricks.com/ppai-capstone-poc/ppai_poc_capstone_experiment"
mlflow.create_experiment(exp_name)

# activate the new experiment
experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
mlflow.set_experiment(experiment_id=experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # Split data and train model with mlflow logging

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(
    maintenance_hum_df.toPandas().drop("MACHINE_FAILURE", axis=1), maintenance_hum_df.toPandas()["MACHINE_FAILURE"], test_size=0.3
)

# COMMAND ----------



# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
mlflow.autolog()
model = RandomForestClassifier()


# COMMAND ----------

run = mlflow.start_run()

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Quick Model Metrics

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import accuracy_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# COMMAND ----------

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

run = mlflow.get_run(run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # More model metrics

# COMMAND ----------

pd.DataFrame(data=[run.data.params], index=["Value"]).T

# COMMAND ----------

pd.DataFrame(data=[run.data.metrics], index=["Value"]).T

# COMMAND ----------

# evaluate model on test
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
y_pred = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='example estimator')
display.plot()
plt.show()

# COMMAND ----------

# auc score
roc_auc_score(y_test, y_pred)

# COMMAND ----------

from sklearn.inspection import permutation_importance
feature_names = ['AIR_TEMPERATURE_K',
       'PROCESS_TEMPERATURE', 'ROTATIONAL_SPEED_RPM', 'TORQUE_NM',
       'TOOL_WEAR_MIN', 'HUMIDITY_RELATIVE_AVG']
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

forest_importances = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Register training dataset with mlflow runid

# COMMAND ----------

# load the previously trained model and extract training dataset
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

#
# NEED TO FIX PERMISSIONS FOR THE FOLLOWING TO STORE ON CLOUD STORAGE
#


# write the training dataset to Delta Lake table in Databricks
# specify the Delta Lake table path
# delta_table_path = "wasbs://ppai-poc-capstone-storage-container@ppaipoccapstone.blob.core.windows.net/mnt/delta/ppai-poc-capstone-dataset"
# maintenance_hum_df.write.format("delta").mode("overwrite").save(delta_table_path)

#register the saved Delta table as a Databricks table
# spark.sql(f"CREATE TABLE IF NOT EXISTS capstone_training_dataset USING DELTA LOCATION '{delta_table_path}'")


# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model

# COMMAND ----------

model_name = "ppai_poc_capstone_model"
model_registered = mlflow.register_model(f"runs:/{run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy Model to snowflake

# COMMAND ----------


