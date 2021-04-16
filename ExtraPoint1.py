# Extra point 1
import numpy as np
import matplotlib.pyplot as plt
from part1 import show_image, reading_files
from part4 import start_learning_with_vectorization, calculate_accuracy

# initialize parameters
n_h1 = n_h2 = 16  # hidden layer 1, 2
n_x = 784  # size of the input layer
n_y = 10  # size of the output layer
learning_rate, number_of_epochs, batch_size, number_of_images = 1, 5, 50, 60000


# Reading The Train Set, shift 4pixel to right, then return train_set and test_set
def reading_files_and_shift_right_test_set():
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

    shifted_test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        # shifting image by 4 pixels to right
        new_image = image.reshape((28, 28))
        for j in range(4):
            new_image = np.roll(new_image, 1)
            new_image[:, 0] = 0.0
        image = new_image.reshape(784, 1)
        shifted_test_set.append((image, label))

    return train_set, shifted_test_set


# def shift_image_4pixel(image_pixels):
#     for i in range(0, 92):
#         image_pixels = np.insert(image_pixels, 0, 0.0)
#     new_image_pixels = np.reshape(image_pixels[:-92], (784, 1))
#     return new_image_pixels
#
#
# def shift_all_images_4pixel(dataset):
#     for image_index in range(len(dataset)):
#         old_image_pixels = dataset[image_index][0]
#         dataset[image_index][0] = shift_image_4pixel(old_image_pixels)
#     return dataset


if __name__ == "__main__":
    # train_set, test_set = reading_files()
    train_set, shifted_test_set = reading_files_and_shift_right_test_set()
    W1, W2, W3, b1, b2, b3, total_cost_arr_in_batch = start_learning_with_vectorization(train_set, number_of_epochs,
                                                                                        batch_size, learning_rate,
                                                                                        "sigmoid")  # this W,b after learning
    accuracy_in_train_set = calculate_accuracy(W1, W2, W3, b1, b2, b3, train_set, "sigmoid")
    print("Accuracy in train set :", accuracy_in_train_set)
    accuracy_in_test_set = calculate_accuracy(W1, W2, W3, b1, b2, b3, shifted_test_set, "sigmoid")
    print("Accuracy in test set :", accuracy_in_test_set)
    plt.plot(total_cost_arr_in_batch)
    plt.show()
    # show_image(test_set[5][0])
    # plt.show()
    # show_image(shifted_test_set[5][0])
    # plt.show()
