import pandas as pd
import csv

def read_csv_for_images(csv_file):
    df = pd.read_csv(csv_file)
    return df

def write_dataset(data_list, csv_name, label_names):
    with open(csv_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['index', 'image_name', 'class_id', 'class_name'])
        for index, each_item in enumerate(data_list):
            csv_writer.writerow([index] + each_item + [label_names[each_item[-1]]])


#  read training image names
train_csv_file = '/media/faiz/data2/classification/logos/annotations/train_images_root.txt'
df_train = read_csv_for_images(train_csv_file)
df_train = df_train[df_train.keys()[0]].tolist() + [df_train.keys()[0]]

print(len(df_train))

# read test image names
test_csv_file = '/media/faiz/data2/classification/logos/annotations/test_images_root.txt'
df_test = read_csv_for_images(test_csv_file)
df_test = df_test[df_test.keys()[0]].tolist() + [df_test.keys()[0]]

print(len(df_test))

label_names = '/media/faiz/data2/classification/logos/annotations/class_names.txt'
label_names = read_csv_for_images(label_names)
label_names = label_names[label_names.keys()[0]].tolist() + [label_names.keys()[0]]

print(len(label_names))

# adding class_id to category_names
new_train_images = []
for each_image_name in df_train:
    class_name = each_image_name.split('/')[1]
    class_id = label_names.index(class_name)
    new_train_images.append([each_image_name, class_id])

# adding class_id to category_names
new_test_images = []
for each_image_name in df_test:
    class_name = each_image_name.split('/')[1]
    class_id = label_names.index(class_name)
    new_test_images.append([each_image_name, class_id])

print(len(new_train_images), new_train_images[0])
print(len(new_test_images), new_test_images[0])

write_dataset(data_list=new_train_images, csv_name='train.csv', label_names=label_names)
write_dataset(data_list=new_test_images, csv_name='test.csv', label_names=label_names)


