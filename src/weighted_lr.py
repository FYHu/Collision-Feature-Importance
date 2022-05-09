from model_evaluation import f1_score
from sklearn.metrics import accuracy_score, precision_score
from train_test_split import train_test_split
from sklearn.linear_model import LogisticRegression
from time_series_gridSearchCV import my_gridSearchCV
from load_data import load_data
from model_evaluation import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from feature_importance import plot_feature_importance
from cross_entropy import cross_entropy

df = load_data()

y = df.pop('Seriousness')
X = df
X_train, X_test, y_train, y_test = train_test_split(X,y)

weight_candidates = [{0:1000,1:100},{0:1000,1:10}, {0:1000,1:1.0},
     {0:500,1:1.0}, {0:400,1:1.0}, {0:300,1:1.0}, {0:200,1:1.0},
     {0:150,1:1.0}, {0:100,1:1.0}, {0:99,1:1.0}, {0:10,1:1.0},
     {0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100},
     {0:0.001,1:1.0}, {0:0.005,1:1.0}, {0:1.0,1:1.0},
     {0:1.0,1:0.1}, {0:10,1:0.1}, {0:100,1:0.1},
     {0:10,1:0.01}, {0:1.0,1:0.01}, {0:1.0,1:0.001}, {0:1.0,1:0.005},
     {0:1.0,1:10}, {0:1.0,1:99}, {0:1.0,1:100}, {0:1.0,1:150},
     {0:1.0,1:200}, {0:1.0,1:300},{0:1.0,1:400},{0:1.0,1:500},
     {0:1.0,1:1000}, {0:10,1:1000},{0:100,1:1000} ]

grid = {"class_weight": weight_candidates}

# define model
lg = LogisticRegression(random_state=13)

best_par = my_gridSearchCV(df, lg, grid)

best_lg = LogisticRegression(random_state=13, class_weight=best_par["class_weight"])

y_pred = best_lg.predict(X_test)
train_pred = best_lg.predict(X_train)

print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
print(f'Recall score: {precision_score(y_test,y_pred)}')
print(f'F1 Score: {f1_score(y_test,y_pred)}')

print("Cross Entropy Loss for training is:" , cross_entropy(y_train, train_pred))
print("Cross Entropy Loss for testing is:" , cross_entropy(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes = ['Death Occur', 'No Death Occur'],
                      title='Confusion Matrix LR')

importance = best_lg.coef_[0]
# summarize feature importance
for g,s in enumerate(importance):
     print('Feature: %0d, Score: %.5f' % (g,s))


