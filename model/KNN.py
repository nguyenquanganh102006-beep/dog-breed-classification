import numpy as np
import joblib

def euclidean_distance(x1, x2):
    return sum((x1-x2)**2)**0.5
class KNN:
    def __init__(self, k=3, X_train=None, y_train=None):
        self.k=k
        self.X_train=X_train
        self.y_train=y_train
    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train
    def predict(self,X):
        return [self._predict(x) for x in X]
    def _predict(self,x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        last_label = np.bincount(k_nearest_labels).argmax()
        return last_label
def train_KNN(X_train, y_train,X_test, y_test, k=3, device='cpu'):
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)   
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    knn = KNN(k=k).to(device)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    print(f"KNN Accuracy: {(y_pred == y_train).mean() * 100:.2f}%")
    y_pred_test = knn.predict(X_test)
    print(f"KNN Test Accuracy: {(y_pred_test == y_test).mean() * 100:.2f}%")
    wandb.log({
        "KNN Train Accuracy": (y_pred == y_train).mean() * 100, 
        "KNN Test Accuracy": (y_pred_test == y_test).mean() * 100})
    joblib.dump(knn, 'model/knn_model.pkl')
    return knn