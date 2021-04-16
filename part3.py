# part 3
import numpy as np
import matplotlib.pyplot as plt
from part1 import reading_files
from part2 import initialize_W_b, linear_activation_forward, sigmoid, calculate_accuracy, tanh

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer
learning_rate, number_of_epochs, batch_size, number_of_images = 1, 20, 10, 100


def update_grad_W_b_in_layer(A3, A2, A1, W3, W2, Z3, Z2, Z1, image, grad_W3, grad_W2, grad_W1, grad_b3, grad_b2,
                             grad_b1, activationType):
    # layer n_y and n_h2
    for j in range(n_y):
        for k in range(n_h2):
            grad_W3[j][k] += (2 * (A3[j][0] - image[1][j])) * derivative_activation(Z3[j][0], activationType) * (A2[k][0])
            grad_b3[j][0] += (2 * (A3[j][0] - image[1][j])) * derivative_activation(Z3[j][0], activationType)

    grad_A2 = np.zeros((n_h2, 1))
    for k in range(n_h2):
        for j in range(n_y):
            grad_A2[k][0] += (2 * (A3[j][0] - image[1][j])) * derivative_activation(Z3[j][0], activationType) * W3[j][k]

    # layer n_h2 and n_h1
    for j in range(n_h2):
        for k in range(n_h1):
            grad_W2[j][k] += derivative_activation(Z2[j][0], activationType) * (grad_A2[j][0]) * A1[k][0]
            grad_b2[j][0] += derivative_activation(Z2[j][0], activationType) * (grad_A2[j][0])

    grad_A1 = np.zeros((n_h1, 1))
    for k in range(n_h1):
        for j in range(n_h2):
            grad_A1[k][0] += W2[j][k] * derivative_activation(Z2[j][0], activationType) * grad_A2[j][0]

    # layer n_h1 and n_x
    for j in range(n_h1):
        for k in range(n_x):
            grad_W1[j][k] += derivative_activation(Z1[j][0], activationType) * grad_A1[j][0] * image[0][k]
            grad_b1[j][0] += derivative_activation(Z1[j][0], activationType) * grad_A1[j][0]

    return grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1


# update W and b (end of batch)
def update_W_b(learning_rate, grad_W, W, grad_b, b, batch_size):
    W -= (learning_rate * (grad_W / batch_size))
    b -= (learning_rate * (grad_b / batch_size))
    return W, b


# cost function calculator
def calculate_cost(A3, y):
    cost = 0
    for i in range(len(A3)):
        cost += pow((A3[i] - y[i]), 2)
    return cost


# give array and batch_size => return all batches
def get_all_batches(batch_size, arr):
    arr_size = len(arr)
    batches = []
    for i in range(0, arr_size, batch_size):
        batch = arr[i: i + batch_size]
        batches.append(batch)
    return batches


# derivative of sigmoid
def derivative_activation(x, activationType):
    if activationType == "sigmoid":
        res = (sigmoid(x) * (1 - sigmoid(x)))
    if activationType == "tanh":
        res = (1 - (tanh(x) * tanh(x)))
    return res

# initialize grad_W matrix and vector grad_b for each layer and initialize to 0
def initialize_grad_W_b():
    grad_W1 = np.zeros((n_h1, n_x))
    grad_b1 = np.zeros((n_h1, 1))
    grad_W2 = np.zeros((n_h2, n_h1))
    grad_b2 = np.zeros((n_h2, 1))
    grad_W3 = np.zeros((n_y, n_h2))
    grad_b3 = np.zeros((n_y, 1))
    return grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3


def start_learning_without_vectorization(dataset, number_of_epochs, batch_size, learning_rate, activationType):
    W1, W2, W3, b1, b2, b3 = initialize_W_b()
    total_cost_arr_in_batch = []
    for n in range(0, number_of_epochs):
        new_train_set = dataset
        np.random.shuffle(new_train_set)
        batches = get_all_batches(batch_size, new_train_set)

        for batch in batches:
            total_cost_in_batch = 0
            grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = initialize_grad_W_b()

            for image in batch:
                Z1, A1 = linear_activation_forward(image[0], W1, b1, activationType)
                Z2, A2 = linear_activation_forward(A1, W2, b2, activationType)
                Z3, A3 = linear_activation_forward(A2, W3, b3, activationType)
                total_cost_in_batch += calculate_cost(A3, image[1])

                # backpropagation calculating
                grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = update_grad_W_b_in_layer(A3, A2, A1,
                                                                                                W3,
                                                                                                W2, Z3, Z2,
                                                                                                Z1, image,
                                                                                                grad_W3,
                                                                                                grad_W2,
                                                                                                grad_W1,
                                                                                                grad_b3,
                                                                                                grad_b2,
                                                                                                grad_b1, activationType)

            total_cost_arr_in_batch.append(total_cost_in_batch)
            # update Wi and bi (end of each batch)
            W3, b3 = update_W_b(learning_rate, grad_W3, W3, grad_b3, b3, batch_size)
            W2, b2 = update_W_b(learning_rate, grad_W2, W2, grad_b2, b2, batch_size)
            W1, b1 = update_W_b(learning_rate, grad_W1, W1, grad_b1, b1, batch_size)

    return W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch


if __name__ == "__main__":
    train_set, test_set = reading_files()
    W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch = start_learning_without_vectorization(train_set[0: number_of_images], number_of_epochs, batch_size, learning_rate, "sigmoid")  # this W,b after learning
    accuracy = calculate_accuracy(W1, W2, W3, b1, b2, b3, train_set[0: number_of_images], "sigmoid")
    print("Accuracy :", accuracy)
    plt.plot(total_cost_arr_in_batch)
    plt.show()
