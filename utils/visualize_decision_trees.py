from sklearn.tree import export_graphviz
from io import StringIO  # Python 3.x 中使用 io.StringIO
import pydotplus
from utils.state_manger import state_manger
from PIL import Image as PILImage


def visualize_decision_trees(state: state_manger, output_path: str):
    # 初始化io
    dot_data = StringIO()

    # 處理視覺化資料
    export_graphviz(state.decision_tree,
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
    img = PILImage.open(output_path)
    img.show()
