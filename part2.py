# part 2
import math

import numpy as np
import matplotlib.pyplot as plt
from part1 import reading_files

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer


# initialize W matrix and vector b for each layer
def initialize_W_b():
    W1 = np.random.randn(n_h1, n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2)
    b3 = np.zeros((n_y, 1))
    return W1, W2, W3, b1, b2, b3


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


# this function calculate next A with A_prev, W and b
def linear_activation_forward(A_prev, W, b, activationType):
    if activationType == "sigmoid":
        # Z = np.dot(W, A_prev) + b
        Z = (W @ A_prev) + b
        A = sigmoid(Z)
    if activationType == "tanh":
        Z = (W @ A_prev) + b
        A = tanh(Z)
    return Z, A


# check the guess label with original label, if correct return 1, else return 0
def check_guess_label(guess_label_index, image):
    is_correct = 0
    if image[1][guess_label_index] == 1:
        is_correct = 1
    return is_correct


def calculate_accuracy(W1, W2, W3, b1, b2, b3, dataset, activationType):
    number_of_correct_answer = 0
    for image in dataset:
        Z1, A1 = linear_activation_forward(image[0], W1, b1, activationType)
        Z2, A2 = linear_activation_forward(A1, W2, b2, activationType)
        Z3, A3 = linear_activation_forward(A2, W3, b3, activationType)
        max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
        number_of_correct_answer += check_guess_label(max_value_index_in_A3, image)

    accuracy = (number_of_correct_answer / len(dataset)) * 100
    return accuracy


if __name__ == "__main__":
    train_set, test_set = reading_files()
    number_of_images = 100
    new_train_set = train_set[0: number_of_images]
    W1, W2, W3, b1, b2, b3 = initialize_W_b()
    accuracy = calculate_accuracy(W1, W2, W3, b1, b2, b3, new_train_set, "sigmoid")
    print("Accuracy : ", accuracy)
