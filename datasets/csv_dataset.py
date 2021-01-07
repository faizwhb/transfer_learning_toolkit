import torch
from torch.utils.data import Dataset
import csv
import os
import PIL
import numpy as np
import pandas as pd
from PIL import Image

class Dataset_from_CSV(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.transform = transform
        self.read_data_from_csv(root, csv_file)

    def read_data_from_csv(self, root, file_path):
        self.im_paths = []
        self.ys = []
        self.I = []
        df = pd.read_csv(file_path)
        for id, value in enumerate(df['image_name']):
            image_name = value
            class_id = df['class_id'][id]
            image_path = os.path.join(root, image_name)

            self.im_paths.append(image_path)
            self.ys.append(class_id)
            self.I.append(id)

    def __len__(self):
        return len(self.I)

    def __getitem__(self, index):
        try:
            im = Image.open(self.im_paths[index])
            if len(list(im.split())) == 1: im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
                return im, self.ys[index], index
            else:
                return self.im_paths[index], self.ys[index], index
        except Exception as e:
            print(self.im_paths[index])
            print(e)

    def get_label(self, index):
        return self.ys[index]

    def nb_classes(self):
        return len(set(self.ys))

    def get_class_distribution(self):
        unique_labels, frequencies = np.unique(self.ys, return_counts=True)
        frequencies = 1 - frequencies/len(self.ys)
        inverse_frequencies = frequencies/sum(frequencies)
        return inverse_frequencies.tolist()