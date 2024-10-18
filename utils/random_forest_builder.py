from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from utils.state_manger import state_manger


def random_forest_builder(state: state_manger):
    # 建立隨機森林
    random_forest = RandomForestClassifier(n_estimators=20)
    # 訓練隨機森林
    random_forest.fit(state.X_train, state.y_train)  # train the model
    # 模型預測值
    y_pred = random_forest.predict(state.X_test)  # 模型預測值
    # 配置狀態
    state.set_random_forest(random_forest, y_pred, metrics.accuracy_score(state.y_test, y_pred))
