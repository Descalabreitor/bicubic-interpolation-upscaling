import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import tifffile
import numpy as np


class DataManager:
    def __init__(self, data_directory, lr_dataset_name, hr_dataset_name):
        if os.path.exists(data_directory):
            self.data_folder = data_directory
        else:
            raise FileNotFoundError("Provided data directory does not exist")
        self.lr_dataset_name = lr_dataset_name
        self.hr_dataset_name = hr_dataset_name
        self.data_points = self.get_data_points()

    def get_data_points(self):
        data_points_df = pd.read_csv(self.data_folder + "/metadata.csv", sep=",")
        return data_points_df

    def get_random_data(self, n_samples = 10, n_revisits=1):
        data_names = self.data_points['ID'].sample(n=n_samples)
        hr_images = self.get_hr_images(data_names)
        lr_images = self.get_lr_images(data_names, n_revisits)
        return hr_images, lr_images

    def get_hr_images(self, data_points):
        images = []
        for data_point in data_points:
            directory = self.data_folder + "/" + self.hr_dataset_name + "/" + data_point + "/" + data_point + "_rgb.png"
            images.append(Image.open(directory))
        images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
        return images_tensor

    def get_lr_images(self, data_points, n_revisits=1, image_shape=(157, 159, 3)):
        images = torch.zeros(len(data_points), n_revisits, *image_shape)
        source = "L2A"
        for i, data_point in enumerate(data_points):
            directory = self.data_folder + "/" + self.lr_dataset_name + "/" + data_point + "/"
            lr_images_package = self.read_sat_images(directory, n_revisits=n_revisits)
            images[i] = lr_images_package
        return images

    def transform_to_rgb(self, sat_image):
        blue_band = sat_image[3][:, :, 0]
        green_band = sat_image[3][:, :, 1]
        red_band = sat_image[3][:, :, 2]

        rgb_image = np.zeros((sat_image[3].shape[0], sat_image[3].shape[1], 3))
        rgb_image[:, :, 0] = red_band
        rgb_image[:, :, 1] = green_band
        rgb_image[:, :, 2] = blue_band
        return rgb_image

    def read_sat_images(self, folder, image_shape=(157, 159, 3), n_revisits=1):
        images = torch.zeros(n_revisits, *image_shape)
        i = 1
        for file_name in enumerate(os.listdir(folder)):
            if i > n_revisits:
                break
            if file_name.__contains__("_data"):
                image = tifffile.imread(folder + file_name)
                images[i] = torch.from_numpy(self.transform_to_rgb(image))
                i += 1
        if n_revisits == 1:
            images.squeeze()
        return images
