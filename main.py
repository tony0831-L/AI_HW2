from utils.xgboost_builder import xgboost_builder
from utils.random_forest_builder import random_forest_builder
from utils.visualize_decision_trees import visualize_decision_trees
from utils.decision_tree_builder import decision_tree_builder
from utils.split_data_train import split_data_train
from utils.data_importer import data_importer
from utils.state_manger import state_manger


def main():
    # 初始化狀態管理器
    state = state_manger()
    # 讀取資料
    data_importer("./static/diabetes(1).csv", state)
    # 取得訓練集
    split_data_train(state)

    # 訓練決策樹
    decision_tree_builder(state)
    # 打印準確率
    print("decision_tree_ACC: ", state.decision_tree_Accuracy)
    # 顯示決策樹圖片
    # visualize_decision_trees(state, 'C:/Users/t0983/Downloads/aihw2/src/static/decision_tree.png')

    # 訓練隨機森林
    random_forest_builder(state)
    # 打印準確率
    print("random_forest_ACC: ", state.random_forest_Accuracy)

    # 訓練xgboost
    xgboost_builder(state)
    # 打印準確率
    print("xgboost_ACC: ", state.xgboost_Accuracy)


if __name__ == '__main__':
    main()
