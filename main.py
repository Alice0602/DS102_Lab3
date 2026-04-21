# Import từ các file module do chính bạn viết
from dataset import XRayDataset
from model import SVMSGD
from evaluator import ModelEvaluator

# Import thư viện ngoài cho Assignment 2
from sklearn.linear_model import SGDClassifier

def main():
    # Cấu hình đường dẫn
    TRAIN_DIR = 'chest_xray/train'
    TEST_DIR = 'chest_xray/test'
    
    # 1. TIỀN XỬ LÝ DỮ LIỆU
    print("Đang tải dữ liệu X-Ray (128x128)...")
    dataset = XRayDataset(img_size=128)
    
    X_train, y_train = dataset.load_from_directory(TRAIN_DIR)
    X_test, y_test = dataset.load_from_directory(TEST_DIR)
    
    if len(X_train) == 0:
        print("Lỗi: Không tìm thấy dữ liệu ảnh. Hãy kiểm tra lại đường dẫn.")
        return

    print("Đang chuẩn hóa dữ liệu Z-score...")
    X_train = dataset.fit_transform(X_train)
    X_test = dataset.transform(X_test)
    
    print("-" * 55)

    # ---------------------------------------------------------
    # ASSIGNMENT 1: SVM FROM SCRATCH (NUMPY)
    # ---------------------------------------------------------
    model_scratch = SVMSGD(C=1.0, epoch=300, lr=1e-6)
    model_scratch.fit(X_train, y_train)
    y_pred_scratch = model_scratch.predict(X_test)
    
    ModelEvaluator.print_report(y_test, y_pred_scratch, "ASSIGNMENT 1 (TỰ XÂY DỰNG - NUMPY)")

    # ---------------------------------------------------------
    # ASSIGNMENT 2: SVM LIBRARY (SCIKIT-LEARN)
    # ---------------------------------------------------------
    print("Đang Train SVM (Assigment 2) bằng Scikit-Learn...")
    model_sklearn = SGDClassifier(loss='hinge', max_iter=300, class_weight='balanced', random_state=42)
    model_sklearn.fit(X_train, y_train)
    y_pred_sklearn = model_sklearn.predict(X_test)
    
    ModelEvaluator.print_report(y_test, y_pred_sklearn, "ASSIGNMENT 2 (THƯ VIỆN SCIKIT-LEARN)")

if __name__ == "__main__":
    main()