import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *
import os

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("User-based Validation Accuracy: {}".format(acc))
    return acc

def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    # Transpose matrix to focus on item similarities
    matrix_t = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix_t)
    # Transpose back to original orientation
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item-based Validation Accuracy: {}".format(acc))
    return acc

def find_data_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_dirs = [
        os.path.join(current_dir, '..', 'data'),
        os.path.join(current_dir, 'data'),
        os.path.join(current_dir, '..', '..', 'data'),
    ]
    for dir_path in possible_dirs:
        if os.path.isdir(dir_path) and all(os.path.isfile(os.path.join(dir_path, f)) for f in ['train_sparse.npz', 'valid_data.csv', 'test_data.csv']):
            return dir_path
    return None

def main():
    data_dir = find_data_directory()
    if data_dir is None:
        print("Error: Unable to locate the data directory.")
        print("Please ensure you're running the script from the correct location.")
        return

    print(f"Using data directory: {data_dir}")

    try:
        sparse_matrix = load_train_sparse(data_dir).toarray()
        val_data = load_valid_csv(data_dir)
        test_data = load_public_test_csv(data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Sparse matrix shape:", sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]
    user_based_accuracies = []
    item_based_accuracies = []

    for k in k_values:
        print(f"\nTesting with k = {k}")
        user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        user_based_accuracies.append(user_acc)
        item_based_accuracies.append(item_acc)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, user_based_accuracies, 'bo-', label='User-based')
    plt.plot(k_values, item_based_accuracies, 'ro-', label='Item-based')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('kNN Performance: User-based vs Item-based')
    plt.legend()
    plt.grid(True)
    plt.savefig('knn_performance.png')
    plt.show()

    # Find best k for each method
    best_k_user = k_values[np.argmax(user_based_accuracies)]
    best_k_item = k_values[np.argmax(item_based_accuracies)]

    print(f"\nBest k for User-based: {best_k_user}")
    print(f"Best k for Item-based: {best_k_item}")

    # Evaluate on test data
    print("\nTest Data Performance:")
    knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    knn_impute_by_item(sparse_matrix, test_data, best_k_item)

if __name__ == "__main__":
    main()