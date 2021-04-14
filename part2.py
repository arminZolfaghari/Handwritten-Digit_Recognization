# part 2
import numpy as np
import matplotlib.pyplot as plt

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer


# Reading The Train Set then return train_set and test_set
def reading_files():
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))
        train_set.append((image, label))

    # Reading The Test Set
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))

    return train_set, test_set


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


# this function calculate next A with A_prev, W and b
def linear_activation_forward(A_prev, W, b, activationType):
    if activationType == "sigmoid":
        # Z = np.dot(W, A_prev) + b
        Z = (W @ A_prev) + b
        A = sigmoid(Z)
    return Z, A


# check the guess label with original label, if correct return 1, else return 0
def check_guess_label(guess_label_index, image):
    is_correct = 0
    if image[1][guess_label_index] == 1:
        is_correct = 1
    return is_correct


train_set, test_set = reading_files()
number_of_images = 100
A0 = train_set[0: number_of_images]
W1, W2, W3, b1, b2, b3 = initialize_W_b()

number_of_correct_answer = 0
for i in range(0, number_of_images):
    Z1, A1 = linear_activation_forward(A0[i][0], W1, b1, "sigmoid")
    Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
    Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
    max_value_in_A3 = np.max(A3)
    max_value_index_in_A3 = np.argmax(A3, 0)  # guess label

    number_of_correct_answer += check_guess_label(max_value_index_in_A3, train_set[i])

print("Accuracy :", (number_of_correct_answer / number_of_images) * 100)
