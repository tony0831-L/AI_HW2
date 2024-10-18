from typing import Union, Any

from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class state_manger:
    x: Union[Series, None, DataFrame]
    y: Any

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any

    decision_tree: DecisionTreeClassifier
    decision_tree_y_pred: Any
    decision_tree_Accuracy: float

    random_forest: RandomForestClassifier
    random_forest_y_pred: Any
    random_forest_Accuracy: float

    xgboost_classifier: Any
    xgboost_y_pred: Any
    xgboost_Accuracy: float

    def set_data(self, x: Union[Series, None, DataFrame], y: Any):
        self.x = x
        self.y = y

    def set_train_data(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def set_decision_tree(self, decision_tree: DecisionTreeClassifier, y_pred: Any, Accuracy: float):
        self.decision_tree = decision_tree
        self.decision_tree_y_pred = y_pred
        self.decision_tree_Accuracy = Accuracy

    def set_random_forest(self, random_forest: RandomForestClassifier, y_pred: Any, Accuracy: float):
        self.random_forest = random_forest
        self.random_forest_y_pred = y_pred
        self.random_forest_Accuracy = Accuracy

    def set_xgboost(self, xgboost_classifier: Any, y_pred: Any, Accuracy: float):
        self.xgboost_classifier = xgboost_classifier
        self.xgboost_y_pred = y_pred
        self.xgboost_Accuracy = Accuracy
