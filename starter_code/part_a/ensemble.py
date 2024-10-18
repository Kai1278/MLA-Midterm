import numpy as np
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from scipy import sparse
import os
import sys
import csv

# Utility functions
def load_csv(path):
    if not os.path.exists(path):
        raise Exception(f"The specified path {path} does not exist.")
    data = {"user_id": [], "question_id": [], "is_correct": []}
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data["user_id"].append(int(row["user_id"]))
            data["question_id"].append(int(row["question_id"]))
            data["is_correct"].append(int(row["is_correct"]))
    return data

def load_train_csv(root_dir):
    path = os.path.join(root_dir, "train_data.csv")
    return load_csv(path)

def load_valid_csv(root_dir):
    path = os.path.join(root_dir, "valid_data.csv")
    return load_csv(path)

def load_public_test_csv(root_dir):
    path = os.path.join(root_dir, "test_data.csv")
    return load_csv(path)

def load_train_sparse(root_dir):
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception(f"The specified path {path} does not exist.")
    
    sparse_matrix = sparse.load_npz(path)
    print(f"Loaded sparse matrix with shape: {sparse_matrix.shape}")
    return sparse_matrix

def sigmoid(x):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-x))

# KNN functions
def knn_impute_by_user(matrix, data, k):
    print("Starting KNN imputation by user...")
    imputer = SimpleImputer(strategy='mean')
    matrix_imputed = imputer.fit_transform(matrix)
    
    print("Fitting NearestNeighbors model...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute').fit(matrix_imputed)
    print("Computing nearest neighbors...")
    distances, indices = nbrs.kneighbors(matrix_imputed)
    
    total_users = len(set(data["user_id"]))
    for i, u in enumerate(data["user_id"]):
        if i % 1000 == 0:  # Print progress every 1000 users
            print(f"Processing user {i}/{total_users}")
        similar_users = indices[u][1:]
        sim_scores = 1 - distances[u][1:]
        idx = data["question_id"][i]
        matrix[u, idx] = np.dot(sim_scores, matrix_imputed[similar_users, idx]) / np.sum(sim_scores)
    
    print("KNN imputation by user completed.")
    return matrix

def knn_impute_by_item(matrix, data, k):
    print("Starting KNN imputation by item...")
    matrix = matrix.T
    imputer = SimpleImputer(strategy='mean')
    matrix_imputed = imputer.fit_transform(matrix)
    
    print("Fitting NearestNeighbors model...")
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute').fit(matrix_imputed)
    print("Computing nearest neighbors...")
    distances, indices = nbrs.kneighbors(matrix_imputed)
    
    total_items = len(set(data["question_id"]))
    for i, q in enumerate(data["question_id"]):
        if i % 1000 == 0:  # Print progress every 1000 items
            print(f"Processing item {i}/{total_items}")
        similar_items = indices[q][1:]
        sim_scores = 1 - distances[q][1:]
        idx = data["user_id"][i]
        matrix[q, idx] = np.dot(sim_scores, matrix_imputed[similar_items, idx]) / np.sum(sim_scores)
    
    print("KNN imputation by item completed.")
    return matrix.T

# IRT functions
def irt(data, val_data, lr, iterations, convergence_threshold=1e-5):
    print("Starting IRT...")
    theta = np.random.normal(0, 1, 542)  # Normal initialization
    beta = np.random.normal(0, 1, 1774)  # Normal initialization
    
    for iteration in range(iterations):
        prev_theta = theta.copy()
        prev_beta = beta.copy()
        
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]
            c = data["is_correct"][i]
            p = sigmoid(theta[u] - beta[q])
            theta[u] += lr * (c - p)
            beta[q] -= lr * (c - p)
        
        # Check for convergence
        if np.max(np.abs(theta - prev_theta)) < convergence_threshold and np.max(np.abs(beta - prev_beta)) < convergence_threshold:
            print(f"Convergence reached at iteration {iteration}.")
            break
        
        if iteration % 10 == 0:  # Print progress every 10 iterations
            print(f"IRT Iteration {iteration} completed.")
    
    print("IRT completed.")
    return theta, beta, None

def evaluate(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        p = sigmoid(theta[u] - beta[q])
        pred.append(p >= 0.5)
    return np.mean(pred == data["is_correct"])

# Matrix factorization function
def svd_reconstruct(matrix, k, regularization=1e-5):
    print("Starting SVD reconstruction...")
    try:
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        s = s + regularization  # Regularize singular values
        S = np.zeros((k, k))
        S[:len(s), :len(s)] = np.diag(s[:len(s)])  # Adjust to fit the size of s
        print("SVD reconstruction completed.")
        return U[:, :k], S, Vt[:k, :]
    except np.linalg.LinAlgError as e:
        print(f"SVD did not converge: {e}")
        return None, None, None
    
def create_bootstrap_sample(data):
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])
    
    n_samples = len(user_ids)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    return {
        "user_id": user_ids[indices].tolist(),
        "question_id": question_ids[indices].tolist(),
        "is_correct": is_correct[indices].tolist()
    }

def train_base_models(train_data, val_data, dense_matrix):
    print("Training base models...")
    knn_user = knn_impute_by_user(dense_matrix.copy(), val_data, k=11)
    knn_item = knn_impute_by_item(dense_matrix.copy(), val_data, k=21)
    theta, beta, _ = irt(train_data, val_data, lr=0.01, iterations=100)
    
    U, S, V = svd_reconstruct(dense_matrix, k=100)
    if U is None or S is None or V is None:
        print("Warning: SVD reconstruction failed. Setting V to a zero matrix.")
        V = np.zeros((100, dense_matrix.shape[1]))  # Fallback to zero matrix
    
    print("Base models training completed.")
    return knn_user, knn_item, (theta, beta), V

def ensemble_predict(test_data, base_models, dense_matrix):
    print("Starting ensemble prediction...")
    knn_user, knn_item, (theta, beta), V = base_models
    
    predictions = []
    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        
        knn_user_pred = knn_user[u, q]
        knn_item_pred = knn_item[u, q]
        irt_pred = sigmoid(theta[u] - beta[q])
        
        # Ensure V is valid before using it
        if V is not None and V.shape[0] > 0:
            # Ensure q is within the bounds of V
            if q < V.shape[0]:
                mf_pred = np.dot(dense_matrix[u, :], V.T)[q]
            else:
                mf_pred = 0.5  # Default value if q is out of bounds
        else:
            mf_pred = 0.5  # Default prediction if SVD failed
        
        ensemble_pred = np.mean([knn_user_pred, knn_item_pred, irt_pred, mf_pred])
        
        predictions.append(ensemble_pred >= 0.5)
    
    print("Ensemble prediction completed.")
    return np.array(predictions)

def evaluate_ensemble(predictions, true_labels):
    return np.mean(predictions == true_labels)

def find_data_directory():
    current_dir = os.getcwd()
    possible_dirs = ['data', 'starter_code/data', 'MLA/starter_code/data']

    for _ in range(3):  # Limit the search to 3 levels up
        for dir_name in possible_dirs:
            test_path = os.path.join(current_dir, dir_name)
            if os.path.isdir(test_path):
                return test_path
        current_dir = os.path.dirname(current_dir)

    return None

def main():
    data_dir = find_data_directory()
    if data_dir is None:
        print("Error: Unable to locate the data directory.")
        print("Please ensure you're running the script from the correct location.")
        return
    
    print(f"Looking for data in: {data_dir}")
    
    required_files = ["train_data.csv", "valid_data.csv", "test_data.csv", "train_sparse.npz"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_files:
        print(f"Error: The following required files are missing from {data_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return

    try:
        print("Loading data...")
        train_data = load_train_csv(data_dir)
        val_data = load_valid_csv(data_dir)
        test_data = load_public_test_csv(data_dir)
        sparse_matrix = load_train_sparse(data_dir)
        dense_matrix = sparse_matrix.toarray()
        
        print(f"Dense matrix shape: {dense_matrix.shape}")
        print("Data loading completed.")

        print("Training base models...")
        base_models = []
        for i in range(3):
            print(f"Training base model {i+1}/3")
            bootstrap_sample = create_bootstrap_sample(train_data)
            base_models.append(train_base_models(bootstrap_sample, val_data, dense_matrix))
        
        print("Making predictions...")
        val_predictions = ensemble_predict(val_data, base_models[0], dense_matrix)
        test_predictions = ensemble_predict(test_data, base_models[0], dense_matrix)
        
        val_accuracy = evaluate_ensemble(val_predictions, val_data["is_correct"])
        test_accuracy = evaluate_ensemble(test_predictions, test_data["is_correct"])
        
        print(f"Ensemble Validation Accuracy: {val_accuracy}")
        print(f"Ensemble Test Accuracy: {test_accuracy}")
        
        knn_user, knn_item, (theta, beta), _ = base_models[0]
        
        knn_user_acc = np.mean((knn_user[val_data["user_id"], val_data["question_id"]] >= 0.5) == val_data["is_correct"])
        knn_item_acc = np.mean((knn_item[val_data["user_id"], val_data["question_id"]] >= 0.5) == val_data["is_correct"])
        irt_acc = evaluate(val_data, theta, beta)
        
        print(f"KNN (User-based) Accuracy: {knn_user_acc}")
        print(f"KNN (Item-based) Accuracy: {knn_item_acc}")
        print(f"IRT Accuracy: {irt_acc}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()