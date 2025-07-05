import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import optuna
import mlflow.sklearn
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset,TargetDriftPreset,ClassificationPreset,DataQualityPreset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
import dagshub

import dagshub
dagshub.init(repo_owner='sg296858', repo_name='MLopsProject', mlflow=True)

#Reading the data
df=pd.read_csv("data/sample_data.csv")
X=df.drop(columns=['species'])
y=df['species']

mlflow.set_tracking_uri("https://dagshub.com/sg296858/MLopsProject.mlflow")

#Encoding target variable
df['species']=df['species'].replace({'setosa':'0','versicolor':'1','virginica':'2'})

#Train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=11)

#hyperparameter tuning using optuna
def objective(trial):
    max_depth=trial.suggest_int("max_depth",5,50)
    n_estimators=trial.suggest_int("n_estimators",10,50)

    model=RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=11
    )
    score=cross_val_score(model,X_train,Y_train,cv=5,scoring='accuracy').mean()
    return score

mlflow.set_experiment("Iris data Project prediction")

with mlflow.start_run():
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    study.optimize(objective,n_trials=20)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_value",study.best_value)

    best_model=RandomForestClassifier(**study.best_params,random_state=11)
    best_model.fit(X_train,Y_train)
    y_pred=best_model.predict(X_test)
    accuracy=accuracy_score(y_pred,Y_test)

    mlflow.log_metric("accuracy",accuracy)

    #Data Drift Report generation
    reference_data=pd.DataFrame(X_train,columns=['sepal_length','sepal_width','petal_length','petal_width'])
    reference_data['target']=Y_train
    reference_data['prediction']=best_model.predict(X_train)

    current_data=pd.DataFrame(X_test,columns=['sepal_length','sepal_width','petal_length','petal_width'])
    current_data['target']=Y_test
    current_data['prediction']=best_model.predict(X_test)

    report=Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset(),
        ClassificationPreset()
    ])

    report.run(reference_data=reference_data,current_data=current_data)
    report.save_html("iris_data monitoring_report.html")
    mlflow.log_artifact("iris_data monitoring_report.html")
    mlflow.log_artifact(__file__)

    #Confusion_matrix
    cm=confusion_matrix(Y_test,y_pred)
    plt.figure(figsize=(6,5))
    image=sns.heatmap(cm,annot=True)
    plt.savefig("image.png")
    mlflow.log_artifact("image.png")


    
    # 1) Save the model to disk
    model_path = "rf_model2"
    mlflow.sklearn.save_model(best_model, model_path)

    # 2) Log the entire folder as an artifact
    mlflow.log_artifact(model_path)

    print(accuracy)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
