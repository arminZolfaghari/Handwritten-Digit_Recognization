# part 5
import main as m
import numpy as np
import matplotlib.pyplot as plt

number_of_images, batch_size, learning_rate, number_of_epochs = 60000, 50, 1, 5


def start_learning(W_arr, b_arr):
    for n in range(0, number_of_epochs):
        new_train_set = train_set[0 : number_of_images]
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
                grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = update_grad_W_b_in_layer(A3, A2, A1, W3, W2, Z3,
                                                                                                Z2,
                                                                                                Z1, image, grad_W3,
                                                                                                grad_W2,
                                                                                                grad_W1, grad_b3,
                                                                                                grad_b2,
                                                                                                grad_b1)

            total_cost_arr_in_batch.append(total_cost_in_batch)
            # update Wi and bi (end of each batch)
            W3, b3 = update_W_b(learning_rate, grad_W3, W3, grad_b3, b3, batch_size)
            W2, b2 = update_W_b(learning_rate, grad_W2, W2, grad_b2, b2, batch_size)
            W1, b1 = update_W_b(learning_rate, grad_W1, W1, grad_b1, b1, batch_size)
