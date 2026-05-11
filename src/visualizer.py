# ============================================================
# MODULE 5: VISUALIZER
# Chức năng:
# - Vẽ đồ thị hồi quy tuyến tính
# - Hiển thị điểm dữ liệu thực tế
# - Hiển thị đường dự đoán của mô hình
# - In ra hàm số hồi quy tuyến tính cuối cùng
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """
    Module trực quan hóa kết quả mô hình.

    Module này dùng để:
    - Vẽ đồ thị giữa dữ liệu thực tế và đường hồi quy tuyến tính
    - In ra phương trình hồi quy tuyến tính cuối cùng

    Với bài này, mô hình có dạng:

    y = a * x + b

    Trong đó:
    - x là Số tiền phải trả
    - y là kWh
    - a là hệ số góc
    - b là hệ số chặn
    """

    def show_linear_equation(self, model, feature_columns, target_column):
        """
        In ra hàm số hồi quy tuyến tính cuối cùng.

        Tham số:
        - model: mô hình hồi quy tuyến tính đã huấn luyện
        - feature_columns: tên các biến đầu vào
        - target_column: tên biến cần dự đoán
        """

        coefficients = model.coef_
        intercept = model.intercept_

        print("\n===== HÀM SỐ HỒI QUY TUYẾN TÍNH =====")

        equation = f"{target_column} = "

        for i, column in enumerate(feature_columns):
            coef = coefficients[i]

            if i == 0:
                equation += f"{coef:.6f} * {column}"
            else:
                if coef >= 0:
                    equation += f" + {coef:.6f} * {column}"
                else:
                    equation += f" - {abs(coef):.6f} * {column}"

        if intercept >= 0:
            equation += f" + {intercept:.6f}"
        else:
            equation += f" - {abs(intercept):.6f}"

        print(equation)

        print("\nÝ nghĩa:")
        print("- Hệ số của biến đầu vào cho biết khi biến đó tăng 1 đơn vị thì giá trị dự đoán thay đổi bao nhiêu.")
        print("- Hệ số chặn là giá trị dự đoán khi tất cả biến đầu vào bằng 0.")

        return equation

    def plot_regression_line(self, X, y, model, feature_column, target_column):
        """
        Vẽ đồ thị hồi quy tuyến tính.

        Hàm này phù hợp nhất khi mô hình chỉ có 1 biến đầu vào.
        Trong bài này:
        - Trục X: Số tiền phải trả
        - Trục Y: kWh
        """

        # Lấy một cột đầu vào để vẽ đồ thị
        X_plot = X[[feature_column]]

        # Sắp xếp dữ liệu theo trục X để đường hồi quy không bị gãy
        plot_data = pd.DataFrame({
            feature_column: X_plot[feature_column],
            target_column: y
        })

        plot_data = plot_data.sort_values(by=feature_column)

        X_sorted = plot_data[[feature_column]]
        y_sorted = plot_data[target_column]

        # Dự đoán y theo dữ liệu X đã sắp xếp
        y_pred_sorted = model.predict(X_sorted)

        plt.figure(figsize=(8, 5))

        # Vẽ điểm dữ liệu thực tế
        plt.scatter(X_sorted[feature_column], y_sorted, label="Dữ liệu thực tế")

        # Vẽ đường hồi quy tuyến tính
        plt.plot(X_sorted[feature_column], y_pred_sorted, label="Đường hồi quy tuyến tính")

        plt.xlabel(feature_column)
        plt.ylabel(target_column)
        plt.title("Đồ thị hồi quy tuyến tính")
        plt.legend()
        plt.grid(True)

        plt.show()
