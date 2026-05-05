import numpy as np
import joblib

class Node:
    def __init__(self,feature=None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    def _entropy(self,y):
        if len(y) == 0:
            return 0
        proportion = np.bincount(y) / len(y)
        return -np.sum(proportion * np.log2(proportion + 1e-10))
    def _information_gain(self, y, left_y, right_y):
        entropy_left = self._entropy(left_y)
        entropy_right = self._entropy(right_y)
        return self._entropy(y)-(len(left_y)/len(y)*entropy_left + len(right_y)/len(y)*entropy_right)
    def _best_split(self, X, y):
        max_gain = -1
        best_feature = None 
        best_threshold = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_y = y[X[:, feature]<threshold]
                right_y = y[X[:, feature]>=threshold]
                if (len(left_y) == 0 or len(right_y) == 0):
                    continue
                gain = self._information_gain(y, left_y, right_y)
                if gain > max_gain:
                    max_gain = gain
                    best_feature = feature 
                    best_threshold = threshold
        return best_feature, best_threshold
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(set(y)) == 1:
            return Node(value = np.bincount(y).argmax())
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value = np.bincount(y).argmax())
        left_X = X[X[:, feature] <threshold]
        right_X = X[X[:, feature] >=threshold]
        left_y = y[X[:, feature] <threshold]
        right_y = y[X[:, feature] >=threshold]
        left_subtree = self._build_tree(left_X,left_y,depth=depth+1)
        right_subtree = self._build_tree(right_X,right_y,depth=depth+1)
        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])
    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
def train_decision_tree(X_train, y_train,X_test, y_test, max_depth=3, device='cpu'):
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    tree = DecisionTree(max_depth=max_depth).to(device)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    print(f"Decision Tree Accuracy: {(y_pred == y_train).mean() * 100:.2f}%")
    y_pred_test = tree.predict(X_test)
    print(f"Decision Tree Test Accuracy: {(y_pred_test == y_test).mean() * 100:.2f}%")
    wandb.log({
        "Decision Tree Train Accuracy": (y_pred == y_train).mean() * 100, 
        "Decision Tree Test Accuracy": (y_pred_test == y_test).mean() * 100})
    joblib.dump(tree, 'model/decision_tree_model.pkl')
    return tree