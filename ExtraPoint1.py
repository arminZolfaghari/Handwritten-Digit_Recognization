# Extra point 1
import numpy as np
import matplotlib.pyplot as plt
from part1 import show_image
from part5 import reading_files


def shift_image_4pixel(image_pixels):
    for i in range(0, 92):
        image_pixels = np.insert(image_pixels, 0, 0.0)
    new_image_pixels = np.reshape(image_pixels[:-92], (784, 1))
    return new_image_pixels


def shift_all_images_4pixel(dataset):
    for image_index in range(len(dataset)):
        old_image_pixels = dataset[image_index][0]
        dataset[image_index][0] = np.array(shift_image_4pixel(old_image_pixels))
    return dataset


if __name__ == "__main__":
    train_set, test_set = reading_files()
    shifted_test_set = shift_all_images_4pixel(test_set)
    show_image(test_set[5][0])
    plt.show()
    show_image(shifted_test_set[5][0])
    plt.show()
