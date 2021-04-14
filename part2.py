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
            grad_W3[j][k] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0]) * (A2[k][0])
            grad_b3[j][0] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0])

    grad_A2 = np.zeros((n_h2, 1))
    for k in range(n_h2):
        for j in range(n_y):
            grad_A2[k][0] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0]) * W3[j][k]

    # layer n_h2 and n_h1
    for j in range(n_h2):
        for k in range(n_h1):
            grad_W2[j][k] += sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0]) * A1[k][0]
            grad_b2[j][0] += sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0])

    grad_A1 = np.zeros((n_h1, 1))
    for k in range(n_h1):
        for j in range(n_h2):
            grad_A1[k][0] += W2[j][k] * sigmoid_derivative(Z2[j][0]) * grad_A2[j][0]

    # layer n_h1 and n_x
    for j in range(n_h1):
        for k in range(n_x):
            grad_W1[j][k] += sigmoid_derivative(Z1[j][0]) * grad_A1[j][0] * image[0][k]
            grad_b1[j][0] += sigmoid_derivative(Z1[j][0]) * grad_A1[j][0]

    return grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1


# update W and b (end of batch)
def update_W_b(learning_rate, grad_W, W, grad_b, b, batch_size):
    W -= (learning_rate * (grad_W / batch_size))
    b -= (learning_rate * (grad_b / batch_size))
    return W, b








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