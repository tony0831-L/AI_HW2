from typing import Union, Any

from sklearn.model_selection import train_test_split

from utils.state_manger import state_manger


def split_data_train(state: state_manger):
    # 分割訓練集
    X_train, X_test, y_train, y_test = train_test_split(state.x, state.y, test_size=0.3)
    # 配置狀態
    state.set_train_data(X_train, X_test, y_train, y_test)
