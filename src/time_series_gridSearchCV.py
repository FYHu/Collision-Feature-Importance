from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV






def my_gridSearchCV(df, model, grid):
    best_para = 0
    y = df.pop('Seriousness')
    X = df
    current_score = 0
    tsc = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=4, test_size=None)

    for train_index, test_index in tsc.split(X):
        X_train01, X_test01 = X.iloc[train_index], X.iloc[test_index]
        y_train01, y_test01 = y.iloc[train_index], y.iloc[test_index]
        grid = GridSearchCV(model,grid,scoring="f1", cv=5, n_jobs=-1, refit=True)
        grid.fit(X_train01, y_train01)
        print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')
        if grid.best_score_ > current_score:
            best_para = grid.best_params_



    return best_para