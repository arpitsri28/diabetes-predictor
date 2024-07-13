import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

def train_and_save_model():
    dataset = pd.read_csv('diabetes.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    classifier = LogisticRegression(random_state=0)
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_classifier = grid_search.best_estimator_
    pickle.dump(best_classifier, open('diabetes_model.pkl', 'wb'))

if __name__ == '__main__':
    train_and_save_model()
