from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x):
    """ Apply sigmoid function. """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood. """
    log_lklihood = 0.
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data["is_correct"][i]
        log_lklihood += c * (theta[u] - beta[q]) - np.log(1 + np.exp(theta[u] - beta[q]))
    return -log_lklihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent. """
    d_theta = np.zeros(len(theta))
    d_beta = np.zeros(len(beta))
    
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data["is_correct"][i]
        p = sigmoid(theta[u] - beta[q])
        d_theta[u] += c - p
        d_beta[q] -= c - p
    
    theta += lr * d_theta
    beta += lr * d_beta
    return theta, beta

def irt(data, val_data, lr, iterations):
    """ Train IRT model. """
    theta = np.random.rand(max(set(data["user_id"])) + 1)
    beta = np.random.rand(max(set(data["question_id"])) + 1)

    train_lld_list = []
    val_lld_list = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_lld_list.append(-neg_lld)
        val_lld_list.append(-val_neg_lld)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, train_lld_list, val_lld_list

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy. """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def find_data_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_dirs = [
        os.path.join(current_dir, '..', 'data'),
        os.path.join(current_dir, 'data'),
        os.path.join(current_dir, '..', '..', 'data'),
    ]
    for dir_path in possible_dirs:
        if os.path.isdir(dir_path):
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
        train_data = load_train_csv(data_dir)
        val_data = load_valid_csv(data_dir)
        test_data = load_public_test_csv(data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Tune learning rate and number of iterations
    lr = 0.01
    iterations = 100
    theta, beta, val_acc_lst, train_lld_list, val_lld_list = irt(train_data, val_data, lr, iterations)

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(iterations), train_lld_list, label="Training")
    plt.plot(range(iterations), val_lld_list, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.legend()
    plt.title("Training Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), val_acc_lst)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Iterations")
    plt.tight_layout()
    plt.savefig("irt_training_curve.png")
    plt.show()

    # Report final validation and test accuracy
    val_score = evaluate(val_data, theta, beta)
    test_score = evaluate(test_data, theta, beta)
    print("Final Validation Accuracy: {}".format(val_score))
    print("Final Test Accuracy: {}".format(test_score))

    # Plot probability of correct response vs theta for three questions
    plt.figure(figsize=(10, 5))
    theta_range = np.linspace(-5, 5, 100)
    question_ids = [0, len(beta) // 2, len(beta) - 1]  # Choose three distinct questions
    for q in question_ids:
        probs = sigmoid(theta_range - beta[q])
        plt.plot(theta_range, probs, label=f"Question {q}")
    plt.xlabel("Student Ability (θ)")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.title("Item Characteristic Curves")
    plt.savefig("item_characteristic_curves.png")
    plt.show()

if __name__ == "__main__":
    main()