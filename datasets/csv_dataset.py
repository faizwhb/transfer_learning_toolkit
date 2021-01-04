import torch
from torch.utils.data import Dataset
import csv
import os
import PIL
import numpy as np
import pandas as pd
from PIL import Image

class Dataset_from_CSV(Dataset):
    def __init__(self, root, csv_file, class_subset=None, transform=None):
        self.transform = transform if transform is not None else None
        self.read_data_from_csv(root, csv_file, class_subset)

    def read_data_from_csv(self, root, file_path, class_subset):
        self.im_paths = []
        self.ys = []
        self.I = []
        df = pd.read_csv(file_path)
        for id, value in enumerate(df['img_file']):
            image_name = value
            class_id = df['class_id'][id]
            image_path = os.path.join(root, image_name)

            if class_subset is not None:
                if class_id in class_subset:
                    self.im_paths.append(image_path)
                    self.ys.append(class_id)
                    self.I.append(id)
            else:
                self.im_paths.append(image_path)
                self.ys.append(class_id)
                self.I.append(id)

        #if min(self.ys) is not 0:
        #    ys = [label - min(self.ys) for label in self.ys]
        #   self.ys = ys

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

#val_dataset  = Dataset_from_CSV(root="/media/faiz/data2/object_detection/VOCdevkit/image_blending",
#                csv_file="/media/faiz/data2/object_detection/VOCdevkit/image_blending/lists/real.csv",
#                class_subset=[15, 12, 9],
#                transform=None)

#for path, label, index in val_dataset:
#    print(path, label, index)