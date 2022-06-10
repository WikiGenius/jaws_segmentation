
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


def axial_dataset_train(image_size, validation_ratio=0.1):
    files = glob.glob('dataset/axial/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (JawsDataset(files[validation_files_count:], image_size),
            JawsDataset(files[:validation_files_count], image_size))


def coronal_dataset_train(image_size, validation_ratio=0.1):
    files = glob.glob('dataset/coronal/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (JawsDataset(files[validation_files_count:], image_size),
            JawsDataset(files[:validation_files_count], image_size))


def sagittal_dataset_train(image_size, validation_ratio=0.1):
    files = glob.glob('dataset/sagittal/train/**/*.dicom.npy.gz')
    assert len(files) > 0
    validation_files_count = ceil(len(files) * validation_ratio)

    return (JawsDataset(files[validation_files_count:], image_size),
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
