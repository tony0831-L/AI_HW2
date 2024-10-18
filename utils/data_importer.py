import pandas as pd
from typing import Union
from pandas import DataFrame, Series

from utils.state_manger import state_manger


def data_importer(url: str, state: state_manger):
    # 導入資料
    data = pd.read_csv(url)
    # 指定欄位名
    x: Union[Series, None, DataFrame] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome']]
    # 結果
    y = data.Outcome
    # 配置狀態
    state.set_data(x, y)


