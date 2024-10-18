import os
from typing import Any

from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
from io import StringIO
from xgboost import plot_tree
import pydotplus
from PIL import Image as PILImage


def visualize_decision_trees(model: Any, output_path: str, openPic: bool = False):
    # 初始化io
    dot_data = StringIO()

    # 處理視覺化資料
    export_graphviz(model,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome'],
                    class_names=['0', '1'])

    # 創建圖片
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # 寫入檔案
    graph.write_png(output_path)

    # 打開圖片
    if openPic:
        img = PILImage.open(output_path)
        img.show()


def visualize_xgboost_decision_trees(model: Any, output_path: str, openPic: bool = False):
    for i in range(model.n_estimators):
        plt.figure(figsize=(20, 10))
        plot_tree(model, num_trees=i, show_values=True)

        # 寫入檔案
        plt.savefig(output_path + "xgboost" + str(i) + ".png")
        plt.close()

    # 打開圖片
    if openPic:
        plt.show()
