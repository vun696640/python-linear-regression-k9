# ============================================================
# MODULE 5: VISUALIZER
# Chức năng:
# - Vẽ đồ thị hồi quy tuyến tính
# - Hiển thị các điểm dữ liệu thực tế
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

    Số tiền phải trả = a * kWh + b

    Trong đó:
    - x là kWh
    - y là Số tiền phải trả
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
        print("- Hệ số của kWh cho biết khi điện năng tiêu thụ tăng 1 kWh thì số tiền phải trả thay đổi bao nhiêu.")
        print("- Hệ số chặn là số tiền dự đoán khi kWh bằng 0.")

        return equation

    def plot_regression_line(self, X, y, model, feature_column, target_column):
        """
        Vẽ đồ thị hồi quy tuyến tính.

        Đồ thị hiển thị:
        - Các điểm dữ liệu thực tế
        - Đường hồi quy tuyến tính của mô hình

        Với bài này:
        - Trục X: kWh
        - Trục Y: Số tiền phải trả
        """

        # Kiểm tra cột đầu vào có tồn tại không
        if feature_column not in X.columns:
            print(f"\nKhông tìm thấy cột '{feature_column}' trong dữ liệu X.")
            print("Các cột hiện có:", list(X.columns))
            return

        # Lấy cột đầu vào để vẽ trục X
        X_plot = X[[feature_column]]

        # Tạo DataFrame chứa dữ liệu thực tế
        plot_data = pd.DataFrame({
            feature_column: X_plot[feature_column],
            target_column: y
        })

        # Sắp xếp dữ liệu theo trục X
        plot_data = plot_data.sort_values(by=feature_column)

        # Tách lại dữ liệu sau khi sắp xếp
        X_sorted = plot_data[[feature_column]]
        y_sorted = plot_data[target_column]

        # Dự đoán giá trị y theo các điểm X thực tế
        y_pred_sorted = model.predict(X_sorted)

        # Tạo khung hình
        plt.figure(figsize=(9, 6))

        # Vẽ các điểm dữ liệu thực tế
        plt.scatter(
            X_sorted[feature_column],
            y_sorted,
            label="Điểm dữ liệu thực tế",
            marker="o",
            s=45
        )

        # Vẽ đường hồi quy tuyến tính
        plt.plot(
            X_sorted[feature_column],
            y_pred_sorted,
            label="Đường hồi quy tuyến tính"
        )

        # Ghi tên trục và tiêu đề
        plt.xlabel(feature_column)
        plt.ylabel(target_column)
        plt.title("Đồ thị hồi quy tuyến tính: kWh và số tiền phải trả")

        # Hiển thị chú thích và lưới
        plt.legend()
        plt.grid(True)

        # Lưu ảnh để xem được trong Colab / Codespaces / GitHub
        plt.savefig("regression_chart.png", dpi=300, bbox_inches="tight")

        # Hiển thị đồ thị
        plt.show()

        print("\nĐã vẽ đồ thị hồi quy tuyến tính.")
        print("Đã lưu biểu đồ vào file: regression_chart.png")
