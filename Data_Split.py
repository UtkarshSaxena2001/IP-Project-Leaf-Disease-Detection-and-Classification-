import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(dataset_dir, output_dir, test_size=0.2, random_state=42):
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        file_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
        train_files, val_files = train_test_split(file_paths, test_size=test_size, random_state=random_state)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        for file in train_files:
            shutil.copy(file, os.path.join(train_class_dir, os.path.basename(file)))
        for file in val_files:
            shutil.copy(file, os.path.join(val_class_dir, os.path.basename(file)))

dataset_dir = "/home/utkarsh/Desktop/Image_Processing/Project/archive/PlantVillage"  
split_dir = "/home/utkarsh/Desktop/Image_Processing/Project/Split_data"    
split_dataset(dataset_dir, split_dir, test_size=0.2, random_state=42)
