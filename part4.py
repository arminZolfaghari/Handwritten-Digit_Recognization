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
learning_rate, number_of_epochs, batch_size = 1, 200, 20
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

