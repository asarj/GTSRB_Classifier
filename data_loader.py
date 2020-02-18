import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GTSRB():
    images = list()
    labels = list()

    def __init__(self, dir):
        self.load(dir)

    def load(self, directory:str)->list:
        directories = [d for d in os.listdir(directory)
                       if os.path.isdir(os.path.join(directory, d))]

        for d in directories:
            label_dir = os.path.join(directory, d)
            files = [os.path.join(label_dir, f)
                     for f in os.listdir(label_dir)
                     if f.endswith(".ppm")]

            for f in files:
                self.images.append(f)
                self.labels.append(int(d))

    def display_one(self, a, title1 = "Original"):
        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def display_two(self, a, b, title1="Original", title2="Edited"):
        plt.subplot(121)
        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(b)
        plt.title(title2)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def processing(self, data):
        # loading image
        # Getting 3 images to work with
        img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:3]]
        print('Original size', img[0].shape)
        # --------------------------------
        # setting dim of the resize
        height = 200
        width = 200
        dim = (width, height)
        res_img = []
        for i in range(len(img)):
            res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
            res_img.append(res)

        # Checking the size
        print("RESIZED", res_img[1].shape)

        # Visualizing one of the images in the array
        original = res_img[1]
        self.display_one(original)

if __name__ == "__main__":
    train_path = "GTSRB/Final_Training/Images"
    test_path = "GTSRB/Final_Test/Images"
    gtsrb = GTSRB(train_path)
    gtsrb.processing(gtsrb.images)