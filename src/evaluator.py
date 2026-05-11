# ============================================================
# MODULE 3: MODEL EVALUATOR
# Chức năng:
# - Đánh giá mô hình hồi quy tuyến tính
# - Tính MAE, MSE, RMSE, R2 Score
# - Hiển thị bảng so sánh thực tế và dự đoán
# ============================================================

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """
    Module đánh giá chất lượng mô hình.

    Các chỉ số:
    - MAE: sai số tuyệt đối trung bình
    - MSE: sai số bình phương trung bình
    - RMSE: căn bậc hai của MSE
    - R2 Score: mức độ mô hình giải thích được dữ liệu
    """

    def evaluate(self, y_test, predictions):
        """
        Tính và in ra các chỉ số đánh giá mô hình.
        """

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n===== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH =====")
        print("MAE  - Sai số tuyệt đối trung bình:", mae)
        print("MSE  - Sai số bình phương trung bình:", mse)
        print("RMSE - Căn bậc hai của MSE:", rmse)
        print("R2 Score - Mức độ giải thích dữ liệu:", r2)

        return mae, mse, rmse, r2

    def show_predictions(self, y_test, predictions):
        """
        Hiển thị bảng so sánh giá trị thực tế và giá trị dự đoán.
        """

        result = pd.DataFrame({
            "Thực tế": y_test.values,
            "Dự đoán": predictions,
            "Sai số": y_test.values - predictions
        })

        print("\n===== SO SÁNH THỰC TẾ VÀ DỰ ĐOÁN =====")
        print(result)

        return result
