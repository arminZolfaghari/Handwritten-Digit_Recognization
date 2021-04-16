# part 5
import numpy as np
import matplotlib.pyplot as plt
from part1 import reading_files
from part4 import start_learning_with_vectorization, calculate_accuracy

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer
learning_rate, number_of_epochs, batch_size, number_of_images = 1, 5, 50, 60000

if __name__ == "__main__":
    train_set, test_set = reading_files()
    W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch = start_learning_with_vectorization(train_set, number_of_epochs, batch_size, learning_rate, "sigmoid")  # this W,b after learning
    accuracy_in_train_set = calculate_accuracy(W1, W2, W3, b1, b2, b3, train_set, "sigmoid")
    print("Accuracy in train set :", accuracy_in_train_set)
    accuracy_in_test_set = calculate_accuracy(W1, W2, W3, b1, b2, b3, test_set, "sigmoid")
    print("Accuracy in test set :", accuracy_in_test_set)
    plt.plot(total_cost_arr_in_batch)
    plt.show()
