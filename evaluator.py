import numpy as np

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, target_label):
        TP = np.sum((y_true == target_label) & (y_pred == target_label))
        FP = np.sum((y_true != target_label) & (y_pred == target_label))
        FN = np.sum((y_true == target_label) & (y_pred != target_label))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

    @classmethod
    def print_report(cls, y_true, y_pred, model_name="MÔ HÌNH"):
        print(f"\n{'='*55}\n KẾT QUẢ ĐÁNH GIÁ: {model_name}\n{'='*55}")
        
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        prec_neg1, rec_neg1, f1_neg1 = cls.calculate_metrics(y_true, y_pred, target_label=-1)
        prec_1, rec_1, f1_1 = cls.calculate_metrics(y_true, y_pred, target_label=1)

        print(f"Accuracy tổng thể: {accuracy * 100:.2f}%\n")
        print(f"{'Lớp':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 55)
        print(f"{'Normal (-1)':<15} | {prec_neg1:<10.4f} | {rec_neg1:<10.4f} | {f1_neg1:<10.4f}")
        print(f"{'Pneumonia (1)':<15} | {prec_1:<10.4f} | {rec_1:<10.4f} | {f1_1:<10.4f}")
        print("\n")