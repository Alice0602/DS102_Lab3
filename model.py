import numpy as np
from tqdm import tqdm

class SVMSGD:
    def __init__(self, C: float = 1.0, epoch: int = 100, lr: float = 1e-5, random_state: int = 42):
        self.C = C
        self.epoch = epoch
        self.lr = lr
        self.random_state = random_state
        self.w = None
        self.b = 0
        self.losses = [] 

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X)).flatten()

    def hinge_loss(self, y: np.ndarray, y_hat_raw: np.ndarray) -> float:
        delta = 1 - y * y_hat_raw
        hinge_part = np.where(delta > 0, delta, 0).sum()
        l2_part = 0.5 * (self.w.T @ self.w).item()
        return l2_part + self.C * hinge_part

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, dim = X.shape
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.losses = []
        y = y.reshape(-1, 1)

        rng = np.random.RandomState(self.random_state)
        pbar = tqdm(range(self.epoch), desc="Đang Train SVM (From Scratch)", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        
        for epoch in pbar:
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_samples):
                x_i = X_shuffled[i].reshape(-1, 1)
                a_n = y_shuffled[i].item()
                y_i = (self.w.T @ x_i + self.b).item()
                
                current_C = self.C * 3.0 if a_n == -1 else self.C * 1.0
                
                condition = a_n * y_i
                l2_grad = self.w / n_samples
                
                if condition >= 1:
                    self.w -= self.lr * l2_grad
                else:
                    self.w -= self.lr * (l2_grad - current_C * a_n * x_i)
                    self.b -= self.lr * (-current_C * a_n)
            
            y_hat_raw = self.decision_function(X)
            loss = self.hinge_loss(y, y_hat_raw)
            self.losses.append(loss)
            pbar.set_postfix({"loss": f"{loss:.2f}"})