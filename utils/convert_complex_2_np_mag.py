import numpy as np
import os


def convert_complex_to_np(complex_folder, npy_folder):
    image_c = np.load(os.path.join(complex_folder, "im_real.npy")) + np.load(os.path.join(complex_folder, "im_imag.npy")) * 1j
    image_m = np.abs(image_c)

    os.makedirs(npy_folder, exist_ok=True)
    
    res_name = os.path.join(npy_folder, 'im_real.npy')
    np.save(res_name, image_m)

    imag_array = np.zeros_like(image_m)
    res_name = os.path.join(npy_folder, 'im_imag.npy')
    np.save(res_name, imag_array)


def main():
    complex_folder = '../examples/FreeMax_206051_2023-09-30-131154_FID013797_COR_T1_2_complex/'
    npy_folder = '../examples_mag/FreeMax_206051_2023-09-30-131154_FID013797_COR_T1_2_mag/'
    convert_complex_to_np(complex_folder, npy_folder)


if __name__ == "__main__":
    main()