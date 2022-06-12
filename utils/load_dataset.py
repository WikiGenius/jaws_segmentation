
from math import ceil
import torch
import gzip
import numpy as np
import glob
from torchvision import transforms


def transformations(is_label: bool, new_size_hw: tuple):
    interpolation = [transforms.InterpolationMode.BICUBIC,
                     transforms.InterpolationMode.NEAREST]
    dataset_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(new_size_hw, interpolation=interpolation[is_label])])
    return dataset_transforms


def transform_data_entry(dicom_path, new_size_hw: tuple):
    image_transforms = transformations(is_label=False, new_size_hw=new_size_hw)
    label_transforms = transformations(is_label=True, new_size_hw=new_size_hw)
    label_path = dicom_path.replace('.dicom.npy.gz', '.label.npy.gz')

    dicom_file = gzip.GzipFile(dicom_path, 'rb')
    dicom = np.load(dicom_file)
    dicom = dicom.astype('float32')

    dicom = image_transforms(dicom)
    spread = dicom.max() - dicom.min()
    if spread != 0:
        dicom = (dicom - dicom.min()) / spread
    else:
        dicom = torch.zeros(dicom.shape)
    label_file = gzip.GzipFile(label_path, 'rb')
    label = np.load(label_file)
    label = label.astype('float32')
    label = label_transforms(label).long()
    return dicom, label


class JawsDataset(torch.utils.data.Dataset):
    def __init__(self, dicom_file_list, new_size_hw: tuple):
        self.dicom_file_list = dicom_file_list
        self.new_size_hw = new_size_hw

    def __len__(self):
        return len(self.dicom_file_list)

    def __getitem__(self, idx):
        dicom_path = self.dicom_file_list[idx]
        dicom, label = transform_data_entry(
            dicom_path, new_size_hw=self.new_size_hw)

        return {
            'image': dicom.contiguous(),
            'mask': label.contiguous()
        }


def axial_dataset_train(image_size, validation_ratio=0.1, multiple_train_size=3):
    files = glob.glob('dataset/axial/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (TrainJawsDataset(files[validation_files_count:], image_size, multiple_train_size=multiple_train_size),
            JawsDataset(files[:validation_files_count], image_size))


def coronal_dataset_train(image_size, validation_ratio=0.1, multiple_train_size=3):
    files = glob.glob('dataset/coronal/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (TrainJawsDataset(files[validation_files_count:], image_size, multiple_train_size=multiple_train_size),
            JawsDataset(files[:validation_files_count], image_size))


def sagittal_dataset_train(image_size, validation_ratio=0.1, multiple_train_size=3):
    files = glob.glob('dataset/sagittal/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (TrainJawsDataset(files[validation_files_count:], image_size, multiple_train_size=multiple_train_size),
            JawsDataset(files[:validation_files_count], image_size))


def axial_dataset_test(image_size):
    files = glob.glob('dataset/axial/test/**/*.dicom.npy.gz')
    assert len(files) > 0
    return JawsDataset(files, image_size)


def coronal_dataset_test(image_size):
    files = glob.glob('dataset/coronal/test/**/*.dicom.npy.gz')
    assert len(files) > 0
    return JawsDataset(files, image_size)


def sagittal_dataset_test(image_size):
    files = glob.glob('dataset/sagittal/test/**/*.dicom.npy.gz')
    assert len(files) > 0
    return JawsDataset(files, image_size)


def transformations_Augmented(is_label: bool, new_size_hw: tuple):
    interpolation = [transforms.InterpolationMode.BICUBIC,
                     transforms.InterpolationMode.NEAREST]
    if not is_label:
        color_transformation = transforms.Compose([transforms.ColorJitter(
            brightness=(0.5, 1.3), contrast=(0.6, 1.5))])
    else:
        color_transformation = None

    center = (new_size_hw[1]//2, new_size_hw[0]//2)
    deformation_transformation = transforms.RandomAffine(
        degrees=30, translate=(0.1, 0.1), shear=10, center=center)

    zoom_in_transformation = transforms.RandomResizedCrop(new_size_hw,
                                                          scale=(0.3, 1.0), ratio=(0.75, 1.3333333333333333),
                                                          interpolation=interpolation[is_label])

    transformations = []

    transformations.append(deformation_transformation)
    transformations.append(zoom_in_transformation)
    dataset_transforms = transforms.Compose(transformations)
    return dataset_transforms, color_transformation


class TrainJawsDataset(torch.utils.data.Dataset):
    def __init__(self, dicom_file_list, new_size_hw: tuple, multiple_train_size=3):
        self.dicom_file_list = dicom_file_list
        self.new_size_hw = new_size_hw
        self.multiple_train_size = multiple_train_size
        self.new_size_dataset = self.multiple_train_size * \
            len(self.dicom_file_list)

        self.image_transforms, self.color_transformation = transformations_Augmented(
            is_label=False, new_size_hw=new_size_hw)
        self.label_transforms, _ = transformations_Augmented(
            is_label=True, new_size_hw=new_size_hw)

    def __len__(self):
        return self.new_size_dataset

    def __getitem__(self, idx):
        if idx >= self.new_size_dataset:
            raise IndexError(
                f"index {idx} is out of bounds for dimension 0 with size {self.new_size}")

        # make idx valid within the data and the size you need
        new_idx = idx % len(self.dicom_file_list)

        # Normal transformations
        dicom_path = self.dicom_file_list[new_idx]
        dicom, label = transform_data_entry(
            dicom_path, new_size_hw=self.new_size_hw)


        if idx >= len(self.dicom_file_list):
            label = label.float()
            # addition transformations for variations (augmnted)

            # make a seed with numpy generator
            seed = np.random.randint(2147483647)
            np.random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7

            dicom = self.image_transforms(dicom)
            dicom = self.color_transformation(dicom)
            np.random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            label = self.label_transforms(label)

            label = label.long()

            spread = dicom.max() - dicom.min()
            if spread != 0:
                dicom = (dicom - dicom.min()) / spread
            else:
                dicom = torch.zeros(dicom.shape)

        return {
            'image': dicom.contiguous(),
            'mask': label.contiguous()
        }
