# ============================================================
# MODULE 2: LINEAR REGRESSION MODEL
# Chức năng:
# - Chia dữ liệu thành tập huấn luyện và tập kiểm tra
# - Khởi tạo mô hình hồi quy tuyến tính
# - Huấn luyện mô hình
# - Dự đoán kết quả
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    """
    Module huấn luyện mô hình hồi quy tuyến tính.

    Hồi quy tuyến tính được dùng để dự đoán một giá trị số liên tục.
    Trong bài này, mô hình dự đoán lượng điện tiêu thụ kWh.
    """

    def __init__(self):
        """
        Khởi tạo mô hình Linear Regression.
        """

        self.model = LinearRegression()

    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành tập huấn luyện và tập kiểm tra.

        test_size = 0.2 nghĩa là:
        - 80% dữ liệu dùng để huấn luyện
        - 20% dữ liệu dùng để kiểm tra

        random_state giúp kết quả chia dữ liệu ổn định hơn.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình bằng dữ liệu training.
        """

        self.model.fit(X_train, y_train)

        print("\nHuấn luyện mô hình thành công.")

        return self.model

    def predict(self, X_test):
        """
        Dự đoán kết quả trên dữ liệu kiểm tra.
        """

        predictions = self.model.predict(X_test)

        return predictions
