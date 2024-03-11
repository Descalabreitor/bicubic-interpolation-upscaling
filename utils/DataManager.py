import os
import random

import pandas as pd
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import tifffile


class DataManager:
    def __init__(self, data_directory, lr_dataset_name, hr_dataset_name):
        if os.path.exists(data_directory):
            self.data_folder = data_directory
        else:
            raise FileNotFoundError("Provided data directory does not exist")
        self.lr_dataset_name = lr_dataset_name
        self.hr_dataset_name = hr_dataset_name
        self.data_points = self.get_data_points()
        self.lr_data_part_names = [
            "CLM.tiff",
            "CLP.tiff",
            "L2A_data.tiff",
            "dataMask.tiff",
            "sunAzimuthAngles.tiff",
            "sunZenithAngles.tiff",
            "viewAzimuthMean.tiff",
            "viewZenithMean.tiff"
        ]

    def get_data_points(self):
        hr_data_points_dir = os.listdir(self.data_folder + "/" + self.hr_dataset_name)
        lr_data_points_dir = os.listdir(self.data_folder + "/" + self.lr_dataset_name)
        data_points_dir = list(set(hr_data_points_dir + lr_data_points_dir))

        if len(data_points_dir) != len(hr_data_points_dir):
            raise Exception("Hr and Lr datasets do not match")

        data_points_df = pd.DataFrame()
        data_points_df['points_name'] = data_points_dir  # Dada que los datos son una combinación de dataframes podría
        # poner una opción para decidir con que dataframes se trabaja.
        return data_points_df

    def get_random_data(self, n_samples):
        data_names = self.data_points['points_name'].sample(n=n_samples)
        hr_images = self.get_hr_images(data_names)
        lr_images = self.get_lr_images(data_names)
        return hr_images, lr_images

    def get_hr_images(self, data_points):
        images = []
        for data_point in data_points:
            directory = self.data_folder + "/" + self.hr_dataset_name + "/" + data_point + "/" + data_point + "_rgb.png"
            images.append(Image.open(directory))
        images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
        return images_tensor

    def get_lr_images(self, data_points):
        images_tensor = torch.zeros((len(data_points)))
        source = "L2A"  # Con el core es fijo aqui pero en el  futuro debe ser interchangeable
        for i, data_point in enumerate(data_points):
            data_directory = self.data_folder + "/" + self.lr_dataset_name + "/" + data_point + "/" + source + "/"
            image_id = random.randint(1, 8)
            parts_list = []
            for data_part_name in self.lr_data_part_names:
                directory = data_directory + data_point + "-" + str(image_id) + "-" + source + data_part_name
                parts_list.append(transforms.ToTensor()(tifffile.imread(directory)))
            image_tensor = torch.stack(parts_list)
            images_tensor[i] = image_tensor
        return images_tensor
