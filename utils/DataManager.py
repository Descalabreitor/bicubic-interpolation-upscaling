import os
import pandas as pd


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
        data_points_df['points_name'] = data_points_dir # Dada que los datos son una combinación de dataframes podría
        # poner una opción para decidir con que dataframes se trabaja.
        return data_points_df

    def get_random_data(self, n_samples): #Probablemente habrá que cambiar esto para que vaya con tensores para el tema de la gpu
        data_names = self.data_points['points_name'].sample(n=n_samples)
        hr_images = list()
        lr_images = list()
        for data_name in data_names:
            hr_images.append(self.get_hr_image(self.data_folder + "/" + self.hr_dataset_name + "/" + data_name))
            lr_images.append(self.get_lr_image(self.data_folder + "/" + self.lr_dataset_name + "/" + data_name))

    def get_hr_image(self, dir):
        return 1
    def get_lr_image(self, dir):
        return 1