#part 3
W1, W2, W3, b1, b2, b3 = initialize_W_b(n_x, n_h1, n_h2, n_y)
learning_rate, number_of_epochs, batch_size, number_of_images = 1, 20, 10, 100
number_of_correct_answer = 0
total_cost_arr_in_batch = []

for n in range(0, number_of_epochs):
    new_train_set = train_set[0:number_of_images]
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

            # backpropagation calculating
            grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = update_grad_W_b_in_layer(A3, A2, A1, W3, W2, Z3, Z2,
                                                                                               Z1, image, grad_W3, grad_W2,
                                                                                               grad_W1, grad_b3, grad_b2,
                                                                                               grad_b1)

            # layer n_y and n_h2
            # for j in range(n_y):
            #     for k in range(n_h2):
            #         grad_W3[j][k] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0]) * (A2[k][0])
            #         grad_b3[j][0] += (2 * (A3[j][0] - image[1][j])) * sigmoid_derivative(Z3[j][0])
            #
            # grad_A2 = np.zeros((n_h2, 1))
            # for k in range(n_h2):
            #     for j in range(n_y):
            #         grad_A2[k][0] += (2 * (A3[j][0] - 2 * image[1][j])) * sigmoid_derivative(Z3[j][0]) * W3[j][k]
            #
            # # layer n_h2 and n_h1
            # for j in range(n_h2):
            #     for k in range(n_h1):
            #         grad_W2[j][k] += sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0]) * A1[k][0]
            #         grad_b2[j][0] += sigmoid_derivative(Z2[j][0]) * (grad_A2[j][0])
            #
            # grad_A1 = np.zeros((n_h1, 1))
            # for k in range(n_h1):
            #     for j in range(n_h2):
            #         grad_A1[k][0] += W2[j][k] * sigmoid_derivative(Z2[j][0]) * grad_A2[j][0]
            #
            # # layer n_h1 and n_x
            # for j in range(n_h1):
            #     for k in range(n_x):
            #         grad_W1[j][k] += sigmoid_derivative(Z1[j][0]) * grad_A1[j][0] * image[0][k]
            #         grad_b1[j][0] += sigmoid_derivative(Z1[j][0]) * grad_A1[j][0]

        total_cost_arr_in_batch.append(total_cost_in_batch)
        # update Wi and bi (end of each batch)
        W3, b3 = update_W_b(learning_rate, grad_W3, W3, grad_b3, b3, batch_size)
        W2, b2 = update_W_b(learning_rate, grad_W2, W2, grad_b2, b2, batch_size)
        W1, b1 = update_W_b(learning_rate, grad_W1, W1, grad_b1, b1, batch_size)


for n in range(0, number_of_epochs):
    new_train_set = train_set[0:number_of_images]
    # np.random.shuffle(new_train_set)
    batches = get_all_batches(batch_size, new_train_set, 100)

    for batch in batches:
        total_cost_in_batch = 0
        # grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = initialize_grad_W_b(n_x, n_h1, n_h2, n_y)

        for image in batch:
            Z1, A1 = linear_activation_forward(image[0], W1, b1, "sigmoid")
            Z2, A2 = linear_activation_forward(A1, W2, b2, "sigmoid")
            Z3, A3 = linear_activation_forward(A2, W3, b3, "sigmoid")
            # total_cost_in_batch += calculate_cost(A3, image[1])

            max_value_index_in_A3 = np.argmax(A3, 0)  # guess label
            number_of_correct_answer += check_guess_label(max_value_index_in_A3, image)


print("Accuracy :", (number_of_correct_answer/ (number_of_epochs * number_of_images)) * 100)
plt.plot(total_cost_arr_in_batch)
plt.show()
exit()