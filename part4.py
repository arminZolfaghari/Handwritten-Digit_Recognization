# part 4
import numpy as np
import matplotlib.pyplot as plt
from part1 import reading_files
from part2 import initialize_W_b, linear_activation_forward, calculate_accuracy
from part3 import update_W_b, calculate_cost, get_all_batches, derivative_activation, initialize_grad_W_b

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer
learning_rate, number_of_epochs, batch_size, number_of_images = 1, 200, 20, 100


def update_grad_W_b_in_layer_vectorized(A3, A2, A1, W3, W2, Z3, Z2, Z1, image, grad_W3, grad_W2, grad_W1, grad_b3,
                                        grad_b2,
                                        grad_b1, activationType):

    grad_W3 += (2 * derivative_activation(Z3, activationType) * (A3 - image[1])) @ (np.transpose(A2))
    grad_b3 += (2 * derivative_activation(Z3, activationType) * (A3 - image[1]))
    grad_A2 = np.zeros((n_h2, 1))

    grad_A2 = np.transpose(W3) @ (2 * derivative_activation(Z3, activationType) * (A3 - image[1]))

    grad_W2 += (derivative_activation(Z2, activationType) * grad_A2) @ (np.transpose(A1))
    grad_b2 += (derivative_activation(Z2, activationType) * grad_A2)
    grad_A1 = np.zeros((n_h1, 1))
    grad_A1 = np.transpose(W2) @ (derivative_activation(Z2, activationType) * grad_A2)

    grad_W1 += (derivative_activation(Z1, activationType) * grad_A1) @ (np.transpose(image[0]))
    grad_b1 += (derivative_activation(Z1, activationType) * grad_A1)

    return grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1


def start_learning_with_vectorization(dataset, number_of_epochs, batch_size, learning_rate, activationType):
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
                grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = update_grad_W_b_in_layer_vectorized(A3, A2, A1,
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
    W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch = start_learning_with_vectorization(train_set[0: number_of_images], number_of_epochs, batch_size, learning_rate, "sigmoid")  # this W,b after learning
    accuracy = calculate_accuracy(W1, W2, W3, b1, b2, b3, train_set[0: number_of_images], "sigmoid")
    print("Accuracy :", accuracy)
    plt.plot(total_cost_arr_in_batch)
    plt.show()
