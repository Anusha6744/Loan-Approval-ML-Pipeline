from sklearn.model_selection import GridSearchCV
from pipeline import build_pipeline

def tune_hyperparameters(x, y):
    pipeline=build_pipeline()
    

    param_grid = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [5, 10, None]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(x, y)
    return grid.best_estimator_, grid.best_params_
