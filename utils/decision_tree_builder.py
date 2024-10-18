from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from utils.state_manger import state_manger


def decision_tree_builder(state: state_manger):

    # 建立決策樹
    decision_tree = DecisionTreeClassifier(criterion="entropy")  # create a decision
    # 訓練決策樹
    decision_tree.fit(state.X_train, state.y_train)
    # 模型預測值
    y_pred = decision_tree.predict(state.X_test)
    # 配置狀態
    state.set_decision_tree(decision_tree, y_pred, metrics.accuracy_score(state.y_test, y_pred))
