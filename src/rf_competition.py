from train_test_split import train_test_split
from imblearn.over_sampling import SMOTENC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from load_data import load_data
from sklearn.metrics import confusion_matrix
from model_evaluation import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = load_data()

y = df.pop('Seriousness')
X = df

# Check all find the column name of the categorical column
df_cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols = list(set(df_cols) - set(num_cols))


X_train, X_test, y_train, y_test = train_test_split(X,y,0.1)



# Data Scaling
scaler = StandardScaler()
scaled_num_feats_train = pd.DataFrame(scaler.fit_transform(X_train[num_cols]),
                                     columns=num_cols, index= X_train.index)

for col in num_cols:
    X_train[col] = scaled_num_feats_train[col]
scaled_num_feats_test = pd.DataFrame(scaler.transform(X_test[num_cols]),
                                    columns=num_cols, index= X_test.index)
for col in num_cols:
    X_test[col] = scaled_num_feats_test[col]


#SMOTENC with Random Froest
sm = SMOTENC(categorical_features=cat_cols, random_state=123, sampling_strategy=.6)
X_trainnc, y_trainnc = sm.fit_resample(X_train, y_train)

SMOTE_SRF = RandomForestClassifier(n_estimators=10, random_state=10)
SMOTE_SRF.fit(X_trainnc, y_trainnc)
y_pred_smote = SMOTE_SRF.predict(X_test)
cm_smote = confusion_matrix(y_test, y_pred_smote)
plot_confusion_matrix(cm_smote, classes = ['Death Occur', 'No Death Occur'],
                      title='Confusion Matrix')


#Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=10, random_state=10)
brf.fit(X_train, y_train)
y_pred_brf = brf.predict(X_test)
cm_brf = confusion_matrix(y_test, y_pred_brf)
plot_confusion_matrix(cm_brf, classes = ['Death Occur', 'No Death Occur'],
                      title='Confusion Matrix')

#Normal RandomForest
rf = RandomForestClassifier(n_estimators=10, random_state=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_brf)
plot_confusion_matrix(cm_rf, classes = ['Death Occur', 'No Death Occur'],
                      title='Confusion Matrix')