import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.sparse import random, save_npz, load_npz

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

# Autoencoder model definition
class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        hidden = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(hidden))
        return out

# Training function
def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Create a NaN mask using PyTorch
            nan_mask = torch.isnan(train_data[user_id])  # Get a boolean mask directly from the tensor

            # Use the mask to index into the target and output tensors
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += lamb * model.get_weight_norm()

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f}\t Valid Acc: {valid_acc:.6f}")

    return valid_acc


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

# Main function to run the training
def main():
    base_path = "../data"
    generate_train_sparse(base_path)  # Generate the train_sparse.npz file
    
    zero_train_matrix, train_matrix, valid_data, test_data = load_data(base_path)

    # Hyperparameters
    k_values = [10, 50, 100, 200]
    lambda_values = [0.001, 0.01, 0.1, 1]
    lr = 0.01
    num_epoch = 20

    best_k = None
    best_lambda = None
    best_valid_acc = 0

    # Hyperparameter tuning
    for k in k_values:
        for lamb in lambda_values:
            print(f"Training with k={k}, lambda={lamb}")
            model = AutoEncoder(train_matrix.shape[1], k)
            valid_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_k = k
                best_lambda = lamb

    print(f"Best hyperparameters: k={best_k}, lambda={best_lambda}")
    print(f"Best validation accuracy: {best_valid_acc:.6f}")

    # Train final model with best hyperparameters
    final_model = AutoEncoder(train_matrix.shape[1], best_k)
    train(final_model, lr, best_lambda, train_matrix, zero_train_matrix, valid_data, num_epoch)

    # Evaluate on the test set
    test_acc = evaluate(final_model, zero_train_matrix, test_data)
    print(f"Final test accuracy: {test_acc:.6f}")

if __name__ == "__main__":
    main()
