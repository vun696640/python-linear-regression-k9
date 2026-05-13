# ============================================================
# MODULE 6: MONTHLY ELECTRICITY PREDICTOR
# Chức năng:
# - Đọc dữ liệu tiêu thụ điện trung bình theo tháng
# - Dùng hồi quy tuyến tính để dự đoán số kWh theo tháng
# - Vẽ đồ thị xu hướng tiêu thụ điện theo tháng
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


class MonthlyElectricityPredictor:
    """
    Module dự đoán số điện theo tháng.

    Trong module này:
    - X là Tháng
    - y là AVG, tức số kWh trung bình của 10 hộ gia đình trong từng tháng
    """

    def import_monthly_file(self, file_path):
        """
        Đọc file CSV dữ liệu trung bình theo tháng.
        """

        data = pd.read_csv(file_path, header=None)

        print("\nĐã import dữ liệu tiêu thụ điện theo tháng.")
        print("Đường dẫn file:", file_path)
        print("Kích thước dữ liệu:", data.shape)

        return data

    def clean_monthly_data(self, data):
        """
        Làm sạch dữ liệu theo tháng.

        File mẫu có dạng:
        - Dòng 0: tiêu đề bảng
        - Dòng 1: dòng trống
        - Dòng 2: tên cột
        - Từ dòng 3 trở đi: dữ liệu tháng 1 đến tháng 12
        """

        print("\n===== DỮ LIỆU THÁNG BAN ĐẦU =====")
        print(data.head())

        # Lấy dòng số 2 làm tên cột
        data.columns = data.iloc[2]

        # Lấy dữ liệu từ dòng số 3 trở đi
        data = data.iloc[3:].reset_index(drop=True)

        # Đổi tên cột đầu tiên thành Tháng
        data = data.rename(columns={data.columns[0]: "Tháng"})

        # Chỉ lấy 2 cột cần dùng: Tháng và AVG
        data = data[["Tháng", "AVG"]]

        # Chuyển dấu phẩy thập phân thành dấu chấm
        data["AVG"] = data["AVG"].astype(str).str.replace(",", ".", regex=False)

        # Chuyển dữ liệu sang dạng số
        data["Tháng"] = pd.to_numeric(data["Tháng"], errors="coerce")
        data["AVG"] = pd.to_numeric(data["AVG"], errors="coerce")

        # Xóa dòng lỗi
        data = data.dropna()

        print("\n===== DỮ LIỆU THÁNG SAU KHI LÀM SẠCH =====")
        print(data)

        return data

    def train_monthly_model(self, data):
        """
        Huấn luyện mô hình dự đoán số kWh trung bình theo tháng.
        """

        X = data[["Tháng"]]
        y = data["AVG"]

        model = LinearRegression()
        model.fit(X, y)

        print("\nHuấn luyện mô hình dự đoán số điện theo tháng thành công.")

        return model

    def show_monthly_equation(self, model):
        """
        In ra hàm hồi quy tuyến tính theo tháng.
        """

        a = model.coef_[0]
        b = model.intercept_

        print("\n===== HÀM DỰ ĐOÁN SỐ ĐIỆN THEO THÁNG =====")

        if b >= 0:
            equation = f"kWh trung bình = {a:.2f} * Tháng + {b:.2f}"
        else:
            equation = f"kWh trung bình = {a:.2f} * Tháng - {abs(b):.2f}"

        print(equation)

        return equation

    def predict_month(self, model):
        """
        Nhập tháng và dự đoán số kWh trung bình.
        """

        print("\n===== DỰ ĐOÁN SỐ ĐIỆN THEO THÁNG =====")

        month = int(input("Nhập tháng cần dự đoán số điện trung bình: "))

        new_data = pd.DataFrame({
            "Tháng": [month]
        })

        prediction = model.predict(new_data)[0]

        prediction_value = round(prediction)

        print("\nTháng cần dự đoán:", month)
        print("Số điện trung bình dự đoán:", f"{prediction_value:,} kWh")

        return prediction_value

    def plot_monthly_trend(self, data, model):
        """
        Vẽ đồ thị xu hướng tiêu thụ điện trung bình theo tháng.
        """

        X = data[["Tháng"]]
        y = data["AVG"]

        y_pred = model.predict(X)

        plt.figure(figsize=(10, 6))

        plt.scatter(
            data["Tháng"],
            y,
            label="kWh trung bình thực tế",
            s=50
        )

        plt.plot(
            data["Tháng"],
            y_pred,
            label="Đường xu hướng dự đoán"
        )

        plt.xlabel("Tháng")
        plt.ylabel("kWh trung bình")
        plt.title("Dự đoán xu hướng tiêu thụ điện trung bình theo tháng")
        plt.xticks(range(1, 13))
        plt.legend()
        plt.grid(True)

        plt.savefig("monthly_prediction_chart.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("\nĐã vẽ đồ thị dự đoán số điện theo tháng.")
        print("Đã lưu biểu đồ vào file: monthly_prediction_chart.png")
