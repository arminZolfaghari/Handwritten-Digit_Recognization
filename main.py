import numpy as np
import matplotlib.pyplot as plt

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# initialize W matrix and vector b for each layer
def initialize_W_b(n_x, n_h1, n_h2, n_y):
    W1 = np.random.randn(n_h1, n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2)
    b3 = np.zeros((n_y, 1))
    return W1, W2, W3, b1, b2, b3


def initialize_grad_W_b(n_x, n_h1, n_h2, n_y):
    grad_W1 = np.zeros((n_h1, n_x))
    grad_b1 = np.zeros((n_h1, 1))
    grad_W2 = np.zeros((n_h2, n_h1))
    grad_b2 = np.zeros((n_h2, 1))
    grad_W3 = np.zeros((n_y, n_h2))
    grad_b3 = np.zeros((n_y, 1))
    return grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def sigmoid_derivative(x):
    return (sigmoid(x) * (1 - sigmoid(x)))


# # this function calculate Z with A_prev, W and b
# def calculate_Z(A_prev, W, b):
#     Z = np.dot(W, A_prev) + b
#     return Z

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


# give array and batch_size => return all batches
def get_all_batches(batch_size, arr, arr_size):
    batches = []
    for i in range(0, arr_size, batch_size):
        batch = arr[i: i + batch_size]
        batches.append(batch)
    return batches


# cost function calculator
def calculate_cost(A3, y):
    cost = 0
    for i in range(len(A3)):
        cost += pow((A3[i] - y[i]), 2)
    return cost


def update_grad_W_b_in_layer(A3, A2, A1, W3, W2, Z3, Z2, Z1, image, grad_W3, grad_W2, grad_W1, grad_b3, grad_b2,
                             grad_b1):
    # layer n_y and n_h2
    for j in range(n_y):
        for k in range(n_h2):
            grad_W3[j][k] = (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0]) * (A2[k][0])
            grad_b3[j][0] = (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0])

    grad_A2 = np.zeros((n_h2, 1))
    for k in range(n_h2):
        for j in range(n_y):
            grad_A2[k][0] = (2 * (A3[j][0] - 2 * image[1][j])) * sigmoid_derivative(Z3[j][0]) * W3[j][k]

    # layer n_h2 and n_h1
    for j in range(n_h2):
        for k in range(n_h1):
            grad_W2[j][k] = sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0]) * A1[k][0]
            grad_b2[j][0] = sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0])

    grad_A1 = np.zeros((n_h1, 1))
    for k in range(n_h1):
        for j in range(n_h2):
            grad_A1[k][0] = W2[j][k] * sigmoid_derivative(Z2[j][0]) * grad_A2[j][0]

    # layer n_h1 and n_x
    for j in range(n_h1):
        for k in range(n_x):
            grad_W1[j][k] = sigmoid_derivative(Z1[j][0]) * grad_A1[j][0] * image[0][k]
            grad_b1[j][0] = sigmoid_derivative(Z1[j][0]) * grad_A1[j][0]

    return grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1


# update W and b (end of batch)
def update_W_b(learning_rate, grad_W, W, grad_b, b, batch_size):
    W -= (learning_rate * (grad_W / batch_size))
    b -= (learning_rate * (grad_b / batch_size))
    return W, b


# part 1
# Reading The Train Set
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


# Plotting an image
# show_image(train_set[5][0])
# plt.show()
# print(len(train_set[5][0]))
# print(train_set[5][0])
# print(train_set[5][1])
# exit()

# # part 2
# A0 = train_set[0:100]
# W1, W2, W3, b1, b2, b3 = initialize_W_b(n_x, n_h1, n_h2, n_y)
#
# number_of_correct_answer = 0
# for i in range(0, 100):
#     Z1, A1 = linear_activation_forward(A0[i][0], W1, b1, "sigmoid")
#     Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
#     Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
#     max_value_in_A3 = np.max(A3)
#     max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
#
#     number_of_correct_answer += check_guess_label(max_value_index_in_A3, train_set[i])
#
#
# print("Accuracy :", number_of_correct_answer / 100)


# part 3
# W1, W2, W3, b1, b2, b3 = initialize_W_b(n_x, n_h1, n_h2, n_y)
# learning_rate, number_of_epochs, batch_size = 1, 20, 10
# number_of_correct_answer = 0
# total_cost_arr_in_batch = []
# for n in range(0, 5):
#     new_train_set = train_set[0:100]
#     # np.random.shuffle(new_train_set)
#     batches = get_all_batches(batch_size, new_train_set, 100)
#
#     for batch in batches:
#         total_cost_in_batch = 0
#         grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = initialize_grad_W_b(n_x, n_h1, n_h2, n_y)
#
#         for image in batch:
#             # print("image ", image)
#             # print("***********************************")
#             # print("image[0] ", image[0])
#             # print("***********************************")
#             # print("image[1] ", image[1])
#             # exit()
#             Z1, A1 = linear_activation_forward(image[0], W1, b1, "sigmoid")
#             Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
#             Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
#             total_cost_in_batch += calculate_cost(A3, image[1])
#
#             max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
#             number_of_correct_answer += check_guess_label(max_value_index_in_A3, image)
#
#             # backpropagation calculating
#             # [grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1] += update_grad_W_b_in_layer(A3, A2, A1, W3, W2, Z3, Z2,
#             #                                                                                    Z1, image, grad_W3, grad_W2,
#             #                                                                                    grad_W1, grad_b3, grad_b2,
#             #                                                                                    grad_b1)
#
#             # layer n_y and n_h2
#             for j in range(n_y):
#                 for k in range(n_h2):
#                     grad_W3[j][k] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0]) * (A2[k][0])
#                     grad_b3[j][0] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0])
#
#             grad_A2 = np.zeros((n_h2, 1))
#             for k in range(n_h2):
#                 for j in range(n_y):
#                     grad_A2[k][0] += (2 * (A3[j][0] - 2 * image[1][j])) * sigmoid_derivative(Z3[j][0]) * W3[j][k]
#
#             # layer n_h2 and n_h1
#             for j in range(n_h2):
#                 for k in range(n_h1):
#                     grad_W2[j][k] += sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0]) * A1[k][0]
#                     grad_b2[j][0] += sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0])
#
#             grad_A1 = np.zeros((n_h1, 1))
#             for k in range(n_h1):
#                 for j in range(n_h2):
#                     grad_A1[k][0] += W2[j][k] * sigmoid_derivative(Z2[j][0]) * grad_A2[j][0]
#
#             # layer n_h1 and n_x
#             for j in range(n_h1):
#                 for k in range(n_x):
#                     grad_W1[j][k] += sigmoid_derivative(Z1[j][0]) * grad_A1[j][0] * image[0][k]
#                     grad_b1[j][0] += sigmoid_derivative(Z1[j][0]) * grad_A1[j][0]
#
#         total_cost_arr_in_batch.append(total_cost_in_batch)
#         # update Wi and bi (end of each batch)
#         W3, b3 = update_W_b(learning_rate, grad_W3, W3, grad_b3, b3, batch_size)
#         W2, b2 = update_W_b(learning_rate, grad_W2, W2, grad_b2, b2, batch_size)
#         W1, b1 = update_W_b(learning_rate, grad_W1, W1, grad_b1, b1, batch_size)
#
# print("Accuracy :", number_of_correct_answer)

# plt.plot(total_cost_arr_in_batch)
# plt.show(total_cost_arr_in_batch)


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


# part 4
W1, W2, W3, b1, b2, b3 = initialize_W_b(n_x, n_h1, n_h2, n_y)
W1_before = W1
W2_before = W2
W3_before = W3
learning_rate, number_of_epochs, batch_size = 1, 200, 10
number_of_correct_answer = 0
total_cost_arr_in_batch = []
for n in range(0, number_of_epochs):
    new_train_set = train_set[0:100]
    np.random.shuffle(new_train_set)
    batches = get_all_batches(batch_size, new_train_set, 100)

    for batch in batches:
        total_cost_in_batch = 0
        grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = initialize_grad_W_b(n_x, n_h1, n_h2, n_y)

        for image in batch:
            Z1, A1 = linear_activation_forward(image[0], W1, b1, "sigmoid")
            Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
            Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
            total_cost_in_batch += calculate_cost(A3, image[1])

            max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
            # number_of_correct_answer += check_guess_label(max_value_index_in_A3, image)

            # backpropagation calculating
            grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = update_grad_W_b_in_layer_vectorized(A3, A2, A1, W3,
                                                                                                       W2, Z3, Z2,
                                                                                                       Z1, image,
                                                                                                       grad_W3, grad_W2,
                                                                                                       grad_W1, grad_b3,
                                                                                                       grad_b2,
                                                                                                       grad_b1)

            # grad_W3 += (2 * sigmoid_derivative(Z3) * (A3 - image[1])) @ (np.transpose(A2))
            # grad_b3 += (2 * sigmoid_derivative(Z3) * (A3 - image[1]))
            # grad_A2 = np.zeros((n_h2, 1))
            # grad_A2 = np.transpose(W3) @ (2 * sigmoid_derivative(Z3) * (A3 - image[1]))
            #
            # grad_W2 += (sigmoid_derivative(Z2) * grad_A2) @ (np.transpose(A1))
            # grad_b2 += (sigmoid_derivative(Z2) * grad_A2)
            # grad_A1 = np.zeros((n_h1, 1))
            # grad_A1 = np.transpose(W2) @ (sigmoid_derivative(Z2) * grad_A2)
            #
            # grad_W1 += (sigmoid_derivative(Z1) * grad_A1) @ (np.transpose(image[0]))
            # grad_b1 += (sigmoid_derivative(Z1) * grad_A1)
            # print("grad_W3 ", grad_W3)
            # print("grad_W2 ", grad_W2)
            # exit()

        total_cost_arr_in_batch.append(total_cost_in_batch)
        # update Wi and bi (end of each batch)
        W3, b3 = update_W_b(learning_rate, grad_W3, W3, grad_b3, b3, batch_size)
        W2, b2 = update_W_b(learning_rate, grad_W2, W2, grad_b2, b2, batch_size)
        W1, b1 = update_W_b(learning_rate, grad_W1, W1, grad_b1, b1, batch_size)
        # W3 -= (learning_rate * (grad_W3 / batch_size))
        # # print("W3 ", W3)
        # # print("**************************")
        # # print("W3 - before W3", W3 - W3_before)
        # # exit()
        # b3 -= (learning_rate * (grad_b3 / batch_size))
        # W2 -= (learning_rate * (grad_W2 / batch_size))
        # b2 -= (learning_rate * (grad_b2 / batch_size))
        # W1 -= (learning_rate * (grad_W1 / batch_size))
        # b1 -= (learning_rate * (grad_b1 / batch_size))
        # print("w1 after - w1 before", W1 - W1_before)
        # print("w2 after - w2 before", W2 - W2_before)
        # print("w3 after - w3 before", W3 - W3_before)


# print("w1 after - w1 before", W1 - W1_before)
# print("w2 after - w2 before", W2 - W2_before)
# print("w3 after - w3 before", W3 - W3_before)
for n in range(0, number_of_epochs):
    new_train_set = train_set[0:100]
    np.random.shuffle(new_train_set)
    batches = get_all_batches(batch_size, new_train_set, 100)

    for batch in batches:
        total_cost_in_batch = 0
        grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = initialize_grad_W_b(n_x, n_h1, n_h2, n_y)

        for image in batch:
            Z1, A1 = linear_activation_forward(image[0], W1, b1, "sigmoid")
            Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
            Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
            total_cost_in_batch += calculate_cost(A3, image[1])

            max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
            number_of_correct_answer += check_guess_label(max_value_index_in_A3, image)



print("Accuracy :", (number_of_correct_answer / (number_of_epochs)))
print(total_cost_arr_in_batch)
plt.plot(total_cost_arr_in_batch)
plt.show()
