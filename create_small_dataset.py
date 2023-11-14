import os
import shutil


def copy_data(image_ids, source, destination):
    
    for img_id in image_ids:
        target_folder_path = os.path.join(destination, img_id)
        if not os.path.isdir(target_folder_path):
            os.mkdir(target_folder_path)
            
        for file_name in os.listdir(os.path.join(source, img_id)):
            # construct full file path
            source_file_path = os.path.join(source, img_id, file_name)
            destination_file_path = os.path.join(target_folder_path, file_name)
            
            if os.path.isfile(source_file_path):
                shutil.copy(source_file_path, destination_file_path)


data_dir = 'coco-records-latest'
train_basename = os.path.join(data_dir, "train")
valid_basename = os.path.join(data_dir, "valid")
test_basename = os.path.join(data_dir, "test")   
    
train_image_ids = os.listdir(train_basename)
valid_image_ids = os.listdir(valid_basename)
test_image_ids = os.listdir(test_basename)

# Divide train - val - test datasets with split ratio 60 - 10 - 30
train_size = int(len(train_image_ids) * 1 / 100)
val_size = int(len(valid_image_ids) * 1 / 100)
test_size = int(len(test_image_ids) * 1 / 100)

print("train size:", train_size)
print("val size:", val_size)
print("test size:", test_size)

train_image_ids = train_image_ids[:train_size]
valid_image_ids = valid_image_ids[:val_size]
test_image_ids = test_image_ids[:test_size]

test_n_images, valid_n_images, train_n_images = len(test_image_ids), len(valid_image_ids), len(train_image_ids)


# Create coco-records-small directory and copy selected subset of the main dataset

target_dir = 'coco-records-very-small'
target_train_basename = os.path.join(target_dir, "train")
target_valid_basename = os.path.join(target_dir, "valid")
target_test_basename = os.path.join(target_dir, "test")

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
    
if not os.path.isdir(target_train_basename):
    os.mkdir(target_train_basename)
                    
if not os.path.isdir(target_valid_basename):
    os.mkdir(target_valid_basename)
    
if not os.path.isdir(target_test_basename):
    os.mkdir(target_test_basename)
    
f = open(os.path.join(target_dir, "data_info.txt"), "w")
f.write("train_n_images: {} \n".format(train_n_images))
f.write("valid_n_images: {} \n".format(valid_n_images))
f.write("test_n_images: {} \n".format(test_n_images))
f.close()


# Copy meta.json and qd_coco_meta.json files
meta_filepath = os.path.join(data_dir, "meta.json")
shutil.copy(meta_filepath, os.path.join(target_dir, "meta.json"))

qd_coco_meta_filepath = os.path.join(data_dir, "qd_coco_meta.json")
shutil.copy(qd_coco_meta_filepath, os.path.join(target_dir, "qd_coco_meta.json"))


# Copy all train - val - test images to the destination

print("Train images are copying now...")
copy_data(train_image_ids, train_basename, target_train_basename)
print("Val images are copying now...")
copy_data(valid_image_ids, valid_basename, target_valid_basename)
print("Test images are copying now...")
copy_data(test_image_ids, test_basename, target_test_basename)



