# ============================================================
# MAIN PROGRAM
# Chức năng:
# - Gọi các module trong thư mục src/
# - Đọc dữ liệu từ thư mục data/
# - Làm sạch dữ liệu
# - Huấn luyện mô hình hồi quy tuyến tính
# - Đánh giá mô hình
# - In hàm số hồi quy tuyến tính
# - Vẽ đồ thị hồi quy tuyến tính
# - Kiểm thử với dữ liệu mới
# ============================================================

from src.data_processor import DataProcessor
from src.linear_model import LinearRegressionModel
from src.evaluator import ModelEvaluator
from src.prediction_tester import PredictionTester
from src.visualizer import Visualizer


def main():
    """
    Hàm chính điều khiển toàn bộ chương trình.

    Quy trình:
    1. Đọc file dữ liệu
    2. Làm sạch dữ liệu
    3. Tách biến đầu vào và biến cần dự đoán
    4. Chia dữ liệu train/test
    5. Huấn luyện mô hình hồi quy tuyến tính
    6. Dự đoán trên tập kiểm tra
    7. Đánh giá mô hình
    8. In hàm số hồi quy tuyến tính
    9. Vẽ đồ thị hồi quy tuyến tính
    10. Kiểm thử với dữ liệu mới
    """

    # Khởi tạo các module
    data_processor = DataProcessor()
    regression_model = LinearRegressionModel()
    evaluator = ModelEvaluator()
    tester = PredictionTester()
    visualizer = Visualizer()

    # Đường dẫn file dữ liệu trong project GitHub
    file_path = "data/energy_training.csv"

    # Cột cần dự đoán
    # Mục tiêu đúng: dùng kWh để dự đoán số tiền phải trả
    target_column = "Số tiền phải trả"

    # Bước 1: Import dữ liệu
    data = data_processor.import_file(file_path)

    # Bước 2: Làm sạch dữ liệu
    data = data_processor.clean_data(data)

    # Bước 3: Hiển thị các cột sau khi làm sạch
    print("\nCác cột dùng trong mô hình:")
    print(list(data.columns))

    # Bước 4: Tách dữ liệu đầu vào và đầu ra
    # X sẽ là các cột còn lại, trong bài này nên là kWh
    # y sẽ là Số tiền phải trả
    X, y = data_processor.split_features_target(data, target_column)

    # Bước 5: Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = regression_model.split_train_test(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Bước 6: Huấn luyện mô hình
    model = regression_model.train(X_train, y_train)

    # Bước 7: Dự đoán trên tập kiểm tra
    predictions = regression_model.predict(X_test)

    # Bước 8: Đánh giá mô hình
    evaluator.evaluate(y_test, predictions)

    # Bước 9: Hiển thị bảng so sánh kết quả
    evaluator.show_predictions(y_test, predictions)

    # Bước 10: In ra hàm số hồi quy tuyến tính cuối cùng
    visualizer.show_linear_equation(
        model=model,
        feature_columns=X.columns,
        target_column=target_column
    )

    # Bước 11: Vẽ đồ thị hồi quy tuyến tính
    # Trục X: kWh
    # Trục Y: Số tiền phải trả
    visualizer.plot_regression_line(
        X=X,
        y=y,
        model=model,
        feature_column="kWh",
        target_column=target_column
    )

    # Bước 12: Kiểm thử với dữ liệu mới
    # Người dùng nhập kWh, mô hình dự đoán số tiền phải trả
    tester.test_new_data(model, X.columns)


if __name__ == "__main__":
    main()
