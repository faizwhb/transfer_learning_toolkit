import csv
import os
import glob

def get_class_names(directory):
    classes = []
    for each in os.walk(directory):
        # print(each)
        classes.append(os.path.basename(each[0]))
    return classes[1:]

def get_files_list_for_dataset(root_folder, train_folder, train_names):
    image_list_with_class_id = []
    for index, item in enumerate(train_names):
        path_to_class_dir = os.path.join(train_folder, item)
        file_list = glob.glob(path_to_class_dir + '/*')
        for each in file_list:
            image_list_with_class_id.append([each[len(root_folder)+1:], index])
    return image_list_with_class_id

def write_csv_file_for_list(train_list, csv_file_name):
    counter=0
    with open(csv_file_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['img_file', 'label_id'])
        for each in train_list:
            writer.writerow([each[0], each[1]])

root_dataset_folder = '/media/faiz/data3/text_classifier_dataset/images'
train_folder = '/media/faiz/data3/text_classifier_dataset/images/train'
test_folder = '/media/faiz/data3/text_classifier_dataset/images/test'

train_names = get_class_names(train_folder)
print(train_names)
test_names = get_class_names(test_folder)
#
train_images_list = get_files_list_for_dataset(root_dataset_folder, train_folder, train_names)
print(train_images_list[0])

test_images_list = get_files_list_for_dataset(root_dataset_folder, test_folder, test_names)
print(test_images_list[0])

write_csv_file_for_list(train_list=train_images_list, csv_file_name='train.csv')
write_csv_file_for_list(train_list=test_images_list, csv_file_name='test.csv')