import os
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

    def get_data_points(self):
        hr_data_points_dir = os.listdir(self.data_folder + "/" + self.hr_dataset_name)
        lr_data_points_dir = os.listdir(self.data_folder + "/" + self.lr_dataset_name)
        data_points_dir = set(hr_data_points_dir + lr_data_points_dir)

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

    def get_hr_images(self, data_points):
        images = []
        for data_point in data_points:
            directory = self.data_folder + "/" + self.hr_dataset_name + "/" + data_point + "/" + data_point + "_rgb.png"
            images.append(Image.open(directory))
        images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
        return images_tensor

    def get_lr_images(self, data_points):
        images = torch.zeros(len(data_points))
        source = "L2A"  # Con el core es fijo aqui pero en el  futuro debe ser interchangeable
        for i, data_point in enumerate(data_points):
            directory = self.data_folder + "/" + self.lr_dataset_name + "/" + data_point + "/"
            image_package = []
            for image_id in range(1, 16):
                directory = directory + "-" + str(id) + "-" + source + "_data.tiff"
                image_package.append(tifffile.imread(directory))
            tensor_package = torch.stack([transforms.ToTensor()(img) for img in image_package])
            images[i] = tensor_package
        return images
