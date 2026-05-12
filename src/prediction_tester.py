# ============================================================
# MODULE 4: PREDICTION TESTER
# Chức năng:
# - Kiểm thử mô hình với dữ liệu mới
# - Cho phép người dùng nhập giá trị đầu vào
# - Trả về kết quả dự đoán
# ============================================================

import pandas as pd


class PredictionTester:
    """
    Module kiểm thử mô hình với dữ liệu mới.

    Sau khi mô hình được huấn luyện, người dùng nhập số kWh.
    Mô hình sẽ dự đoán số tiền phải trả tương ứng.
    """

    def test_new_data(self, model, feature_columns):
        """
        Nhập dữ liệu mới và dự đoán kết quả.

        Với bài này:
        - Biến đầu vào: kWh
        - Biến cần dự đoán: Số tiền phải trả
        """

        print("\n===== KIỂM THỬ VỚI DỮ LIỆU MỚI =====")

        new_data = {}

        for column in feature_columns:
            value = float(input(f"Nhập giá trị cho {column}: "))
            new_data[column] = [value]

        new_df = pd.DataFrame(new_data)

        prediction = model.predict(new_df)

        # Làm tròn kết quả dự đoán đến hàng đồng
        prediction_value = round(prediction[0])

        print("\nDữ liệu mới:")
        print(new_df)

        print("\nKết quả dự đoán số tiền phải trả:", f"{prediction_value:,} đồng")

        return prediction_value
