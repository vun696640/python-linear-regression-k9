# ============================================================
# MODULE 1: DATA PROCESSOR
# Chức năng:
# - Đọc dữ liệu từ file CSV trong thư mục data/
# - Làm sạch dữ liệu
# - Xóa cột không cần thiết
# - Chuyển dữ liệu về dạng số
# - Tách dữ liệu thành X và y
# ============================================================

import pandas as pd


class DataProcessor:
    """
    Module xử lý dữ liệu đầu vào.

    Trong project này, dữ liệu được lưu trong file:
    data/energy_training.csv

    Dữ liệu gồm các cột:
    - #: số thứ tự
    - kWh: điện năng tiêu thụ
    - Số tiền phải trả: tiền điện phải trả

    Cột # chỉ là số thứ tự nên sẽ bị loại bỏ.
    """

    def import_file(self, file_path):
        """
        Đọc dữ liệu từ file CSV.

        Tham số:
        - file_path: đường dẫn tới file dữ liệu

        Kết quả:
        - Trả về dữ liệu dạng DataFrame
        """

        data = pd.read_csv(file_path)

        print("Đã import dữ liệu thành công.")
        print("Đường dẫn file:", file_path)
        print("Kích thước dữ liệu:", data.shape)

        return data

    def clean_data(self, data):
        """
        Làm sạch dữ liệu.

        Các bước:
        - Xóa cột # nếu có
        - Chuyển dữ liệu sang dạng số
        - Xóa các dòng bị thiếu hoặc không hợp lệ
        - Kiểm tra dữ liệu có ít nhất 20 mẫu
        """

        print("\n===== DỮ LIỆU BAN ĐẦU =====")
        print(data.head())

        # Xóa khoảng trắng thừa trong tên cột
        data.columns = data.columns.str.strip()

        # Xóa cột số thứ tự nếu tồn tại
        if "#" in data.columns:
            data = data.drop(columns=["#"])

        # Kiểm tra đúng cột cần dùng
        required_columns = ["kWh", "Số tiền phải trả"]

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Không tìm thấy cột bắt buộc: {column}")

        # Chỉ giữ đúng 2 cột cần dùng cho mô hình
        # Tránh lấy nhầm cột khác như tháng, hộ gia đình, số thứ tự...
        data = data[required_columns]

        # Chuyển toàn bộ dữ liệu sang dạng số
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # Xóa dòng có giá trị bị thiếu hoặc không hợp lệ
        data = data.dropna()

        # Xóa dữ liệu vô lý
        data = data[data["kWh"] > 0]
        data = data[data["Số tiền phải trả"] > 0]

        print("\n===== DỮ LIỆU SAU KHI LÀM SẠCH =====")
        print(data.head())

        print("\nSố dòng dữ liệu sau khi làm sạch:", len(data))

        if len(data) < 20:
            raise ValueError("Dữ liệu sau khi làm sạch phải có ít nhất 20 mẫu.")

        return data

    def split_features_target(self, data, target_column):
        """
        Tách dữ liệu thành:
        - X: biến đầu vào
        - y: biến cần dự đoán

        Trong bài này:
        - X là cột kWh
        - y là cột Số tiền phải trả
        """

        if target_column not in data.columns:
            raise ValueError(f"Không tìm thấy cột cần dự đoán: {target_column}")

        # Sửa minimal: chỉ dùng kWh làm biến đầu vào
        # Không dùng data.drop(columns=[target_column]) vì có thể lấy nhầm cột khác
        X = data[["kWh"]]
        y = data[target_column]

        return X, y
