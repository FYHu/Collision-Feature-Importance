# from load_data import load_data
from train_test_split import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from model_evaluation import plot_confusion_matrix
import pandas as pd
from feature_importance import plot_feature_importance
from sklearn.metrics import classification_report
import numpy as np
from temp_data_frame import create_temp_df
from cross_entropy import cross_entropy




df = create_temp_df()



y = df.pop("Seriousness")
X = df.drop("Location", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X,y)
features_to_encode = list(X_train.select_dtypes(include=['object']).columns)
col_trans = make_column_transformer((OneHotEncoder(), features_to_encode), remainder="passthrough")

rf_classifier = RandomForestClassifier(criterion="entropy")

pipe = make_pipeline(col_trans, rf_classifier)
pipe.fit(X_train, y_train).score(X_train, y_train)
y_pred = pipe.predict(X_test)


def encode_and_bind(original_dataframe, features_to_encode):
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([dummies, original_dataframe], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return(res)

X_train_encoded = encode_and_bind(X_train, features_to_encode)


feature_importances = list(zip(X_train_encoded, rf_classifier.feature_importances_))
# Then sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, key=lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked];

print(feature_importances_ranked)

plot_feature_importance(feature_importances_ranked)


train_pred = pipe.predict_proba(X_train)


def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


temp_pred_result = pipe.predict(X_test)
print(classification_report(y_test, temp_pred_result))





#Cross Entropy:
print("Cross Entropy Loss for training is:" , cross_entropy(y_train, train_pred))
print("Cross Entropy Loss for testing is:" , cross_entropy(y_test, y_pred))

print(f"The accuracy of the model is {round(accuracy_score(y_test,y_pred),3)*100} %")
print(f"The precision of the model is {round(precision_score(y_test,y_pred,average='macro'),3)*100} %")
print(f"The recall of the model is {round(recall_score(y_test,y_pred,average='macro'),3)*100} %")


cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes = ['Death Occur', 'No Death Occur'],
                      title='Confusion Matrix RF')