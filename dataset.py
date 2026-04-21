import os
import cv2
import numpy as np

class XRayDataset:
    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        self.mean = None
        self.std = None
        self.categories = {'NORMAL': -1, 'PNEUMONIA': 1}

    def load_from_directory(self, data_dir: str):
        """Quét thư mục, resize, làm phẳng ảnh và gán nhãn"""
        data, labels = [], []
        
        for category, label in self.categories.items():
            folder_path = os.path.join(data_dir, category)
            if not os.path.exists(folder_path): 
                continue
                
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img_resized = cv2.resize(img, (self.img_size, self.img_size))
                    data.append(img_resized.flatten())
                    labels.append(label)
                    
        return np.array(data, dtype=np.float32), np.array(labels)

    def fit_transform(self, X: np.ndarray):
        """Tìm Mean, Std từ tập Train và chuẩn hóa Z-score"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8 
        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray):
        """Dùng Mean, Std đã học để chuẩn hóa tập Test"""
        if self.mean is None or self.std is None:
            raise ValueError("Lỗi: Phải gọi fit_transform() trước!")
        return (X - self.mean) / self.std