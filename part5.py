# part 5
import numpy as np
import matplotlib.pyplot as plt

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer
learning_rate, number_of_epochs, batch_size, number_of_images = 1, 5, 50, 60000


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


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def sigmoid_derivative(x):
    return (sigmoid(x) * (1 - sigmoid(x)))


# initialize W matrix and vector b for each layer
def initialize_W_b():
    W1 = np.random.randn(n_h1, n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2)
    b3 = np.zeros((n_y, 1))
    return W1, W2, W3, b1, b2, b3


# initialize grad_W matrix and vector grad_b for each layer and initialize to 0
def initialize_grad_W_b():
    grad_W1 = np.zeros((n_h1, n_x))
    grad_b1 = np.zeros((n_h1, 1))
    grad_W2 = np.zeros((n_h2, n_h1))
    grad_b2 = np.zeros((n_h2, 1))
    grad_W3 = np.zeros((n_y, n_h2))
    grad_b3 = np.zeros((n_y, 1))
    return grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3


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


def update_grad_W_b_in_layer_vectorized(A3, A2, A1, W3, W2, Z3, Z2, Z1, image, grad_W3, grad_W2, grad_W1, grad_b3,
                                        grad_b2,
                                        grad_b1):
    grad_W3 += (2 * sigmoid_derivative(Z3) * (A3 - image[1])) @ (np.transpose(A2))
    grad_b3 += (2 * sigmoid_derivative(Z3) * (A3 - image[1]))
    grad_A2 = np.zeros((n_h2, 1))
    grad_A2 = np.transpose(W3) @ (2 * sigmoid_derivative(Z3) * (A3 - image[1]))

    grad_W2 += (sigmoid_derivative(Z2) * grad_A2) @ (np.transpose(A1))
    grad_b2 += (sigmoid_derivative(Z2) * grad_A2)
    grad_A1 = np.zeros((n_h1, 1))
    grad_A1 = np.transpose(W2) @ (sigmoid_derivative(Z2) * grad_A2)

    grad_W1 += (sigmoid_derivative(Z1) * grad_A1) @ (np.transpose(image[0]))
    grad_b1 += (sigmoid_derivative(Z1) * grad_A1)

    return grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1


def start_learning():
    W1, W2, W3, b1, b2, b3 = initialize_W_b()
    total_cost_arr_in_batch = []
    for n in range(0, number_of_epochs):
        new_train_set = train_set[0: number_of_images]
        np.random.shuffle(new_train_set)
        batches = get_all_batches(batch_size, new_train_set)

        for batch in batches:
            total_cost_in_batch = 0
            grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = initialize_grad_W_b()

            for image in batch:
                Z1, A1 = linear_activation_forward(image[0], W1, b1, "sigmoid")
                Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
                Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
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
                                                                                                           grad_b1)

            total_cost_arr_in_batch.append(total_cost_in_batch)
            # update Wi and bi (end of each batch)
            W3, b3 = update_W_b(learning_rate, grad_W3, W3, grad_b3, b3, batch_size)
            W2, b2 = update_W_b(learning_rate, grad_W2, W2, grad_b2, b2, batch_size)
            W1, b1 = update_W_b(learning_rate, grad_W1, W1, grad_b1, b1, batch_size)

    return W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch


def calculate_accuracy(W1, W2, W3, b1, b2, b3, dataset):
    number_of_correct_answer = 0
    for image in dataset :
        Z1, A1 = linear_activation_forward(image[0], W1, b1, "sigmoid")
        Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
        max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
        number_of_correct_answer += check_guess_label(max_value_index_in_A3, image)

    accuracy = (number_of_correct_answer / len(dataset)) * 100
    return accuracy



# part 4
train_set, test_set = reading_files()
W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch = start_learning()   #this W,b after learning
accuracy_in_train_set = calculate_accuracy(W1, W2, W3, b1, b2, b3, train_set)
print("Accuracy in train set :", accuracy_in_train_set)
accuracy_in_test_set = calculate_accuracy(W1, W2, W3, b1, b2, b3, test_set)
print("Accuracy in test set :", accuracy_in_test_set)
plt.plot(total_cost_arr_in_batch)
plt.show()
