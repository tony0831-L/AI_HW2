import xgboost
from sklearn import metrics
from utils.state_manger import state_manger


def xgboost_builder(state: state_manger):
    # 建立xgboost分類器
    xgboost_classifier = xgboost.XGBClassifier()
    # 訓練xgboost分類器
    xgboost_classifier.fit(state.X_train, state.y_train)
    # 模型預測值
    y_pred = xgboost_classifier.predict(state.X_test)
    # 配置狀態
    state.set_xgboost(xgboost_classifier, y_pred, metrics.accuracy_score(state.y_test, y_pred))
