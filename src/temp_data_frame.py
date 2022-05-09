import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Temp testing data frame only


d = {'Seriousness': [1,0,0,0,0,0], 'Light Condition': [1, 1,1,1,0,0], 'Road Info': [1, 2,3,4,5,6], 'wind speed':[123,233,232,313,99,132], 'Temperature': [11,33,22,44,55,21],"Cate":["New0", "New1","New2", "New3","New4","New5"]}




def create_temp_df():
    df = pd.DataFrame(data=d)

    return df


# df_temp = create_temp_df()
#
# y = df_temp.pop('Seriousness')
# X = df_temp
#
# tsc = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=4, test_size=None)
#
# for train_index, test_index in tsc.split(X):
#     X_train01, X_test01 = X.iloc[train_index], X.iloc[test_index]
#     y_train01, y_test01 = y.iloc[train_index], y.iloc[test_index]



