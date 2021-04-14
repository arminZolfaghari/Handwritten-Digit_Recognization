W_1 = np.random.standard_normal(size=(16, 784))
b_1 = np.zeros((16, 1))

W_2 = np.random.standard_normal(size=(16, 16))
b_2 = np.zeros((16, 1))

W_3 = np.random.standard_normal(size=(10, 16))
b_3 = np.zeros((10, 1))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cost_calculator(a3, y):
    cost, s = 0, 0
    d1, d2 = a3.shape
    for d in range(d1):
        s = a3[d, 0] - y[d]
        cost += s
    return cost


from random import shuffle

cost_arr = []
for e in range(epoch_num):
    # shuffle(tmp_train_df)
    num_of_batches = int(np.ceil(len(tmp_train_df) / batch_size))
    for batch_index in range(num_of_batches):

        grad_W3 = np.zeros((10, 16))
        grad_b3 = np.zeros((10, 1))

        grad_W2 = np.zeros((16, 16))
        grad_b2 = np.zeros((16, 1))

        grad_W1 = np.zeros((16, 784))
        grad_b1 = np.zeros((16, 1))

        for image_index in range(batch_size * batch_index, batch_size * (batch_index + 1)):

            # forward propagation
            a_1 = feedforward(W_1, tmp_train_df[0][image_index], b_1)
            a_2 = feedforward(W_2, a_1, b_2)
            a_3 = feedforward(W_3, a_2, b_3)

            cost_arr.append(cost_calculator(a_3, tmp_train_df[1][image_index]))

            z_3 = (W_3 @ a_2) + b_3
            z_2 = (W_2 @ a_1) + b_2
            z_1 = (W_1 @ tmp_train_df[0][image_index]) + b_1

            # backpropagation
            # for output layer
            for j in range(10):
                for k in range(16):
                    grad_W3[j, k] += sigmoid_deriv(z_3[j, 0]) * (2 * (a_3[j, 0] - tmp_train_df[1][image_index][j])) * \
                                     a_2[k, 0]
                    grad_b3[j, 0] += sigmoid_deriv(z_3[j, 0]) * (2 * (a_3[j, 0] - tmp_train_df[1][image_index][j]))

            grad_a2 = np.zeros((16, 1))
            for k in range(16):
                for j in range(10):
                    grad_a2[k, 0] += W_3[j, k] * sigmoid_deriv(z_3[j, 0]) * (
                            2 * (a_3[j, 0] - tmp_train_df[1][image_index][j]))

            # for hidden layer 2
            for j in range(16):
                for k in range(16):
                    grad_W2[j, k] += sigmoid_deriv(z_2[j, 0]) * grad_a2[j, 0] * a_1[k, 0]
                    grad_b2[j, 0] += sigmoid_deriv(z_2[j, 0]) * grad_a2[j, 0]

            grad_a1 = np.zeros((16, 1))
            for k in range(16):
                for j in range(16):
                    grad_a1[k, 0] += W_2[j, k] * sigmoid_deriv(z_2[j, 0]) * grad_a2[j, 0]

            # for hidden layer 1
            for j in range(16):
                for k in range(784):
                    grad_W1[j, k] += sigmoid_deriv(z_1[j, 0]) * grad_a1[j, 0] * tmp_train_df[0][image_index][k]
                    grad_b1[j, 0] += sigmoid_deriv(z_1[j, 0]) * grad_a1[j, 0]

        # upgrading W metrices and b vectores
        W_1 -= (alpha * (grad_W1 / batch_size))
        W_2 -= (alpha * (grad_W2 / batch_size))
        W_3 -= (alpha * (grad_W3 / batch_size))
true_ones = 0
for i in range(len(tmp_train_df)):
    if (np.argmax(a_3)) == (np.argmax(tmp_train_df[1][i])):
        true_ones += 1

accuracy = (true_ones / len(tmp_train_df)) * 100
print('Accuracy of our model is:', accuracy)

import matplotlib.pyplot as plt

plt.plot(cost_arr)
plt.show()











# backpropagation
      # for output layer
      for j in range(10):
        for k in range(16):
          grad_W3[j,k] += activation_deriv(z_3[j,0], 'sigmoid') * (2 * (a_3[j,0] - tmp_train_df[1][image_index][j])) * a_2[k,0]
          grad_b3[j,0] += activation_deriv(z_3[j,0], 'sigmoid') * (2 * (a_3[j,0] - tmp_train_df[1][image_index][j]))

      grad_a2 = np.zeros((16,1))
      for k in range(16):
        for j in range(10):
          grad_a2[k,0] += W_3[j,k] * activation_deriv(z_3[j,0], 'sigmoid') * (2 * ( a_3[j,0] - tmp_train_df[1][image_index][j]))

      # for hidden layer 2
      for j in range(16):
        for k in range(16):
          grad_W2[j,k] += activation_deriv(z_2[j,0], 'sigmoid') * grad_a2[j,0] * a_1[k,0]
          grad_b2[j,0] += activation_deriv(z_2[j,0], 'sigmoid') * grad_a2[j,0]

      grad_a1 = np.zeros((16,1))
      for k in range(16):
        for j in range(16):
          grad_a1[k,0] += W_2[j,k] * activation_deriv(z_2[j,0], 'sigmoid') * grad_a2[j,0]


      # for hidden layer 1
      for j in range(16):
        for k in range(784):
          grad_W1[j,k] += activation_deriv(z_1[j,0], 'sigmoid') * grad_a1[j,0] * tmp_train_df[0][image_index][k]
          grad_b1[j,0] += activation_deriv(z_1[j,0], 'sigmoid') * grad_a1[j,0]
