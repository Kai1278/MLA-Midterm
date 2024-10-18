import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.sparse import random, save_npz, load_npz
import matplotlib.pyplot as plt  # Import for visualization

def generate_train_sparse(base_path):
    """Generate and save a random sparse matrix as train_sparse.npz."""
    os.makedirs(base_path, exist_ok=True)
    
    rows = 1000  # Number of users
    cols = 100  # Number of questions
    density = 0.1  # 10% of the matrix will be non-zero
    
    train_sparse = random(rows, cols, density=density, format='csr', dtype=np.float32)
    save_npz(f"{base_path}/train_sparse.npz", train_sparse)
    print(f"train_sparse.npz saved at {base_path}")

def load_train_sparse(base_path):
    """Load the train data as a sparse matrix."""
    train_sparse = load_npz(f"{base_path}/train_sparse.npz")
    return train_sparse

def load_valid_csv(base_path):
    """Generate and return dummy validation data."""
    valid_data = {
        "user_id": np.random.randint(0, 1000, size=500).tolist(),
        "question_id": np.random.randint(0, 100, size=500).tolist(),
        "is_correct": np.random.randint(0, 2, size=500).tolist()
    }
    return valid_data

def load_public_test_csv(base_path):
    """Generate and return dummy public test data."""
    test_data = {
        "user_id": np.random.randint(0, 1000, size=500).tolist(),
        "question_id": np.random.randint(0, 100, size=500).tolist(),
        "is_correct": np.random.randint(0, 2, size=500).tolist()
    }
    return test_data

def load_data(base_path="../data"):
    """Load the data in PyTorch Tensor format."""
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0  # Fill missing values with 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

# Enhanced Autoencoder model definition
class EnhancedAutoEncoder(nn.Module):
    def __init__(self, num_question, k=100, dropout_rate=0.2):
        super(EnhancedAutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.dropout = nn.Dropout(dropout_rate)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        hidden = torch.sigmoid(self.g(inputs))
        hidden = self.dropout(hidden)  # Apply dropout
        out = torch.sigmoid(self.h(hidden))
        return out

# Evaluation function
def evaluate(model, train_data, valid_data):
    model.eval()
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / total

# Training function
def train_with_metadata(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Use Adam optimizer
    num_student = train_data.shape[0]

    for epoch in range(num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = torch.isnan(train_data[user_id])
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += lamb * model.get_weight_norm()

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f}\t Valid Acc: {valid_acc:.6f}")

    return valid_acc

def main_with_metadata():
    base_path = "../data"
    zero_train_matrix, train_matrix, valid_data, test_data = load_data(base_path)

    # Hyperparameters
    k = 100  # Number of hidden units
    lamb = 0.01  # Regularization parameter
    lr = 0.001  # Learning rate
    num_epoch = 20

    print(f"Training Enhanced AutoEncoder with k={k}, lambda={lamb}")
    model = EnhancedAutoEncoder(train_matrix.shape[1], k)
    valid_acc = train_with_metadata(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    print(f"Validation accuracy: {valid_acc:.6f}")

    # Evaluate on the test set
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Final test accuracy: {test_acc:.6f}")

    # Visualization
    # Replace these with the actual baseline accuracy results from Part A
    baseline_valid_accuracy = 0.75  # Example baseline validation accuracy
    baseline_test_accuracy = 0.70    # Example baseline test accuracy

    baseline_results = [baseline_valid_accuracy, baseline_test_accuracy]
    enhanced_results = [valid_acc, test_acc]

    labels = ['Validation Accuracy', 'Test Accuracy']
    x = np.arange(len(labels))

    plt.bar(x - 0.2, baseline_results, 0.4, label='Baseline Model')
    plt.bar(x + 0.2, enhanced_results, 0.4, label='Enhanced Model')

    plt.xlabel('Accuracy Type')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_with_metadata()
