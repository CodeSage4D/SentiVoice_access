# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# import time

# def train_model(X_train, y_train):
#     parameters = {
#         'max_features': ('auto', 'sqrt'),
#         'n_estimators': [100, 200, 300],  # Reduced for faster execution
#         'max_depth': [5, 10, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'bootstrap': [True, False]
#     }

#     grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=3, return_train_score=True, n_jobs=-1)

#     start_time = time.time()
#     grid_search.fit(X_train, y_train)
#     end_time = time.time()

#     elapsed_time = end_time - start_time
#     print(f"Grid Search Elapsed Time: {elapsed_time:.2f} seconds")

#     return grid_search.best_estimator_, grid_search.best_params_


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    nb_model = MultinomialNB()
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0]
    }
    grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=3, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params
