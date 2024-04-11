import numpy as np
import pydicom
import os
import re


def numerical_sort(file_name):
    # Extract numbers from the filename
    numbers = re.findall(r'\d+', file_name)
    # Convert the extracted numbers to integers for sorting
    return [int(num) for num in numbers]

def get_scanlist(dicom_dir):
    all_entries = os.listdir(dicom_dir)
    scanlist = sorted([entry for entry in all_entries if os.path.isdir(os.path.join(dicom_dir, entry))])
    print(">> All scans: ", scanlist)
    return scanlist


def get_subfolfer(dicom_dir):
    all_entries = os.listdir(dicom_dir)
    subfolders = sorted([entry for entry in all_entries if os.path.isdir(os.path.join(dicom_dir, entry))])
    print(">> All subfolders: ", subfolders)
    return subfolders


def get_dicom(subfolder):
    assert os.path.exists(subfolder), "Folder does not exist."
    entries = os.listdir(subfolder)
    assert not any(os.path.isdir(os.path.join(subfolder, entry)) for entry in entries), "Subdirectories found. The folder does not have only one level."
    dicom_names = sorted([entry for entry in entries if os.path.isfile(os.path.join(subfolder, entry))], key=numerical_sort)
    return dicom_names


def convert_dicom_to_np(subfolder_dir, npy_folder):
    dicom_names = get_dicom(subfolder_dir)
    pixel_list = []
    for dicom_name in dicom_names:
        dicom_data = pydicom.dcmread(os.path.join(subfolder_dir, dicom_name))
        pixel_list.append(dicom_data.pixel_array)
    
    first_shape = pixel_list[0].shape
    if not all(inner_list.shape == first_shape for inner_list in pixel_list):
        print("! Image shape not match")
        return

    os.makedirs(npy_folder, exist_ok=True)

    pixel_array = np.transpose(np.array(pixel_list), (1,2,0))
    print(pixel_array.shape)
    res_name = os.path.join(npy_folder, 'im_real.npy')
    np.save(res_name, pixel_array)

    imag_array = np.zeros_like(pixel_array)
    print(imag_array.shape)
    res_name = os.path.join(npy_folder, 'im_imag.npy')
    np.save(res_name, imag_array)


def main():
    dicom_dir = ''
    npy_dir = ''
    os.makedirs(npy_dir, exist_ok=True)
    
    scan_list = get_scanlist(dicom_dir)
    for i, scan_name in enumerate(scan_list):
        scan_dir = dicom_dir + scan_name + '/'
        print("---> Process scan: ", scan_name)
        subfolders = get_subfolfer(scan_dir)

        for j, subfolder in enumerate(subfolders):
            subfolder_dir = scan_dir + subfolder + '/'
            print("---> Process subfolder: ", subfolder)
            npy_folder = os.path.join(npy_dir, scan_name, subfolder)
    
            convert_dicom_to_np(subfolder_dir, npy_folder)


if __name__ == "__main__":
    main()