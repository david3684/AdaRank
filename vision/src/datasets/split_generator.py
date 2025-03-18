import random
from pathlib import Path
import shutil
import os


def process_sun397(src_dir, dst_dir):
    # PROCESS SUN397 DATASET
    downloaded_data_path = f"{src_dir}/sun397"
    output_path = f"{dst_dir}/sun397"

    def process_dataset(txt_file, downloaded_data_path, output_folder):
        with open(txt_file, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            input_path = line.strip()
            final_folder_name = "_".join(
                x for x in input_path.split('/')[:-1])[1:]
            filename = input_path.split('/')[-1]
            output_class_folder = os.path.join(
                output_folder, final_folder_name)

            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            full_input_path = os.path.join(
                downloaded_data_path, input_path[1:])
            output_file_path = os.path.join(output_class_folder, filename)
            # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
            # exit()
            shutil.copy(full_input_path, output_file_path)
            if i % 100 == 0:
                print(f"Processed {i}/{len(lines)} images")

    process_dataset(
        os.path.join(downloaded_data_path, 'Training_01.txt'),
        os.path.join(downloaded_data_path, 'SUN397'),
        os.path.join(output_path, "train")
    )
    process_dataset(
        os.path.join(downloaded_data_path, 'Testing_01.txt'),
        os.path.join(downloaded_data_path, 'SUN397'),
        os.path.join(output_path, "val")
    )


def process_eurosat(src_dir, dst_dir):
    # PROCESS EuroSAT_RGB DATASET
    # base_dir = f'<your base dir>'
    # replace with the path to your dataset
    # src_dir = f'{base_dir}/euro_sat/2750'
    # replace with the path to the output directory
    # dst_dir = f'{base_dir}/EuroSAT_splits'

    def create_directory_structure(dst_dir, classes):
        for dataset in ['train', 'val', 'test']:
            path = os.path.join(dst_dir, dataset)
            os.makedirs(path, exist_ok=True)
            for cls in classes:
                os.makedirs(os.path.join(path, cls), exist_ok=True)

    def split_dataset(dst_dir, src_dir, classes, val_size=270, test_size=270):
        for cls in classes:
            class_path = os.path.join(src_dir, cls)
            images = os.listdir(class_path)
            random.shuffle(images)

            val_images = images[:val_size]
            test_images = images[val_size:val_size + test_size]
            train_images = images[val_size + test_size:]

            for img in train_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(dst_dir, 'train', cls, img)
                print(src_path, dst_path)
                shutil.copy(src_path, dst_path)
                # break
            for img in val_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(dst_dir, 'val', cls, img)
                print(src_path, dst_path)
                shutil.copy(src_path, dst_path)
                # break
            for img in test_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(dst_dir, 'test', cls, img)
                print(src_path, dst_path)
                shutil.copy(src_path, dst_path)
                # break

    classes = [d for d in os.listdir(
        src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    create_directory_structure(dst_dir, classes)
    split_dataset(dst_dir, src_dir, classes)


# PROCESS DTD DATASET
def process_dtd(source_dir, dst_dir):
    downloaded_data_path = f"{source_dir}/dtd/images"
    output_path = f"{dst_dir}/dtd"

    def process_dataset(txt_file, downloaded_data_path, output_folder):
        with open(txt_file, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            input_path = line.strip()
            final_folder_name = input_path.split('/')[:-1][0]
            filename = input_path.split('/')[-1]
            output_class_folder = os.path.join(
                output_folder, final_folder_name)

            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            full_input_path = os.path.join(downloaded_data_path, input_path)
            output_file_path = os.path.join(output_class_folder, filename)
            shutil.copy(full_input_path, output_file_path)
            if i % 100 == 0:
                print(f"Processed {i}/{len(lines)} images")

    process_dataset(
        f'{source_dir}/dtd/labels/train.txt', downloaded_data_path, os.path.join(
            output_path, "train")
    )
    process_dataset(
        f'{source_dir}/dtd/labels/test.txt', downloaded_data_path, os.path.join(
            output_path, "val")
    )


if __name__ == '__main__':
    process_sun397(
        src_dir="/data1/common_datasets/vision_cls/_temp",
        dst_dir="/data1/common_datasets/vision_cls/")