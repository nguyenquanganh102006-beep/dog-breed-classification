import cupy as np
import joblib
import sklearn.preprocessing.OneHotEncoder as OneHotEncoder

class multilayerperception:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, dropout_rate=0):
        self.lr = lr
        self.dropout_rate = dropout_rate
        # Khởi tạo chuẩn theo thứ tự: Layer 1 -> Layer 2 -> Layer 3 (Output)
        # W1: (2, 4)    
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: (4, 4) - Tầng ẩn thứ 2
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b2 = np.zeros((1, hidden_size))
        
        # W3: (4, 1) - Tầng Output
        self.W3 = np.random.randn(hidden_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Trừ max để tránh overflow
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def MSE_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Thêm epsilon để tránh log(0)
        epsilon = 1e-10
        return -np.sum(y_true * np.log(y_pred + epsilon)) / m

    def forward(self, X, is_training=True):
        # L1: X(m,2) . W1(2,4) = (m,4)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # apply dropout for hidden layer 1 during training
        if is_training and self.dropout_rate > 0:
            self.mask1 = (np.random.rand(*self.a1.shape) > self.dropout_rate).astype(float)
            self.a1 *= self.mask1 / (1 - self.dropout_rate)  # Scale up để giữ giá trị trung bình
            
        # L2: a1(m,4) . W2(4,4) = (m,4)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # apply dropout for hidden layer 2 during training
        if is_training and self.dropout_rate > 0:
            self.mask2 = (np.random.rand(*self.a2.shape) > self.dropout_rate).astype(float)
            self.a2 *= self.mask2 / (1 - self.dropout_rate)  # Scale up để giữ giá trị trung bình
        
        # L3: a2(m,4) . W3(4,1) = (m,1)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3) # Dùng Softmax cho Output để ra xác suất 0-1
        return self.a3

    def backward(self, X, y_true):
        m = y_true.shape[0]

        # Đạo hàm tầng 3 (Output)
        dz3 = self.a3 - y_true 
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Đạo hàm tầng 2
        da2 = np.dot(dz3, self.W3.T)
        if self.dropout_rate > 0:
            da2 *= self.mask2 / (1 - self.dropout_rate)  
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Đạo hàm tầng 1
        da1 = np.dot(dz2, self.W2.T)
        if self.dropout_rate > 0:
            da1 *= self.mask1 / (1 - self.dropout_rate)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Cập nhật Gradient (Dùng dấu -=)
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=10000):
        for epoch in range(epochs):
            y_pred = self.forward(X,is_training=True)
            self.backward(X, y)
            if epoch % 1000 == 0:
                acc = (np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)).mean() * 100
                print(f"Epoch {epoch}, Loss: {self.cross_entropy_loss(y, y_pred):.6f}, Accuracy: {acc:.2f}%")
                wandb.log({"epoch": epoch, "loss": self.cross_entropy_loss(y, y_pred), "accuracy": acc})

    def predict(self, X):
        y_pred = self.forward(X,is_training=False)
        # Với XOR 1 output, ta dùng ngưỡng 0.5
        return (y_pred > 0.5).astype(int)
def train_MLP(X_train, y_train,X_test, y_test, hidden_size=8, lr=0.1, epochs=10000, dropout_rate=0, device='cpu'):
    # Encode labels using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False).to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    
    #train MLP
    input_size = X_train.shape[1]
    output_size = y_train_encoded.shape[1]
    mlp = multilayerperception(input_size, hidden_size, output_size, lr, dropout_rate).to(device)
    mlp.fit(X_train, y_train_encoded, epochs)
    y_pred = mlp.predict(X_train)
    print(f"MLP Accuracy: {(y_pred == y_train_encoded).mean() * 100:.2f}%")
    y_pred_test = mlp.predict(X_test)
    print(f"MLP Test Accuracy: {(y_pred_test == y_test_encoded).mean() * 100:.2f}%")
    joblib.dump(mlp, 'model/mlp_model.pkl')
    joblib.dump(encoder, 'model/mlp_encoder.pkl')
    return mlp