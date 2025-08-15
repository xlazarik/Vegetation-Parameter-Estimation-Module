import os
import numpy as np
import pandas
import rasterio


np.seterr(divide='ignore', invalid='ignore')

###################################################################################################
# Search band nearest to the wavelength val of X
# search could be optimized with the use of dict saving already computed
def binary_search(arr, x):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = l + (r - l) // 2
        # Check if x is present at mid
        if arr[mid] == x:
            return mid
            # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1
        # If x is smaller, ignore right half
        else:
            r = mid - 1
    return l if abs(x - arr[l]) < abs(x - arr[r]) else r

###################################################################################################
# Compute individual indices

def compute_aoki(bands):
    aoki_i = np.divide(bands["550"], bands["800"])
    return aoki_i


def compute_datt2(bands):
    datt2_i = np.divide(bands["850"] - bands["710"], bands["850"] - bands["680"])

    return datt2_i


def compute_cri1(bands):
    cri1 = 1 / bands["510"] - 1 / bands["550"]
    return cri1


def compute_cri2(bands):
    cri2_i = 1 / bands["510"] - 1 / bands["700"]
    return cri2_i


def compute_tcari_osavi(bands):
    tcari_osavi_i = 3 * (
            (bands["700"] - bands["670"]) - 0.2 * (bands["700"] - bands["550"]) * (bands["700"] / bands["670"])) / (
                        (1.16 * (bands["800"] - bands["670"]) / (0.16 + bands["800"] + bands["670"])))
    return tcari_osavi_i


def compute_lcai(bands):
    lcai_i = 100 * ((bands["2205"] - bands["2165"]) + (bands["2205"] - bands["2330"]))
    return lcai_i


def compute_sr(bands):
    sr_i = bands["2155"] / bands["1705"]
    return sr_i


def compute_dci(bands):
    dci_i = np.diff(bands["705"]) / np.diff(bands["722"])
    # dci_i[np.where((dci_i < -15) | (dci_i > 15))] = 0
    return np.hstack((dci_i[:, 0].reshape(-1, 1), dci_i))

###################################################################################################
# Index fnc mapping

INDICES = {'aoki': compute_aoki, 'datt2': compute_datt2, 'cri1': compute_cri1, 'cri2': compute_cri2,
           'tcari-osavi': compute_tcari_osavi, 'dci': compute_dci, 'lcai': compute_lcai, 'sr': compute_sr}

###################################################################################################
# Abstract fnc for computation over LUT object

def compute_lut_veg_index_selection(lut_object, resampled_lut_vi_path, hdr_data, required_wavelengths, vi_to_compute,
                                    delimeter=","):
    band_values = {}
    computed_vi = {}

    resampled_file = pandas.read_csv(resampled_lut_vi_path, sep=delimeter, dtype=float, memory_map=True)

    for band in required_wavelengths:
        band_key = hdr_data.wwl_vector[binary_search(hdr_data.wwl_vector, int(band))]
        if band_key in resampled_file:
            band_values[band] = resampled_file[str(band_key)]
        elif str(float(band_key)) in resampled_file:
            band_values[band] = resampled_file[str(float(band_key))]

    for index in vi_to_compute:
        computed_vi[index] = INDICES[index](band_values)

    return computed_vi

###################################################################################################
# Abstract fnc for computation over IMG object

def compute_image_veg_index_selection(image_path, hdr_data, required_wavelengths, vi_to_compute):
    band_values = {}
    computed_vi = {}
    try:
        with rasterio.open(image_path) as dataset:
            for band in required_wavelengths:
                band_values[band] = dataset.read(
                    binary_search(hdr_data.wwl_vector, int(band)) + 1) / hdr_data.reflectance_scale

            for index in vi_to_compute:
                computed_vi[index] = INDICES[index](band_values)
    except IndexError:
        return [], "The image does not contain bands with sufficient wavelengths for the computation of specified vegetation indices"
    return computed_vi, ""


# Computes a selection of all the indices for LUT
def compute_lut_vegetation_indices(lut_path, band_wavelengths, band_labels):
    resampFile = pandas.read_csv(lut_path, sep=',', dtype=float, memory_map=True)
    veg_ind = {'aoki': None, 'datt2': None, 'cri1': None, 'cri2': None, 'tcari_osavi': None}

    # ========================================================================================

    print("WVL NEAREST 550 IS ON INDEX: ", binary_search(band_wavelengths, 550), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 550)])
    band550 = resampFile[band_labels[binary_search(band_wavelengths, 550)]]

    print("WVL NEAREST 800 IS ON INDEX: ", binary_search(band_wavelengths, 800), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 800)])
    band800 = resampFile[band_labels[binary_search(band_wavelengths, 800)]]

    aoki_i = np.divide(band550, band800)
    veg_ind['aoki'] = aoki_i

    print("WVL NEAREST 850 IS ON INDEX: ", binary_search(band_wavelengths, 850), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 850)])
    band850 = resampFile[band_labels[binary_search(band_wavelengths, 850)]]

    print("WVL NEAREST 710 IS ON INDEX: ", binary_search(band_wavelengths, 710), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 710)])
    band710 = resampFile[band_labels[binary_search(band_wavelengths, 710)]]  # 723–885

    print("WVL NEAREST 680 IS ON INDEX: ", binary_search(band_wavelengths, 680), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 680)])
    band680 = resampFile[band_labels[binary_search(band_wavelengths, 680)]]  # 697

    datt2_i = np.divide(band850 - band710, band850 - band680)
    veg_ind['datt2'] = datt2_i

    print("WVL NEAREST 510 IS ON INDEX: ", binary_search(band_wavelengths, 510), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 510)])
    band510 = resampFile[band_labels[binary_search(band_wavelengths, 510)]]

    cri1 = 1 / band510 - 1 / band550
    veg_ind['cri1'] = cri1

    print("WVL NEAREST 700 IS ON INDEX: ", binary_search(band_wavelengths, 700), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 700)])
    band700 = resampFile[band_labels[binary_search(band_wavelengths, 700)]]

    cri2_i = 1 / band510 - 1 / band700
    veg_ind['cri2'] = cri2_i

    print("WVL NEAREST 670 IS ON INDEX: ", binary_search(band_wavelengths, 670), " WITH VALUE ",
          band_wavelengths[binary_search(band_wavelengths, 670)])
    band670 = resampFile[band_labels[binary_search(band_wavelengths, 670)]]

    tcari_osavi_i = 3 * ((band700 - band670) - 0.2 * (band700 - band550) * (band700 / band670)) / (
        (1.16 * (band800 - band670) / (0.16 + band800 + band670)))
    veg_ind['tcari_osavi'] = tcari_osavi_i

    return veg_ind

###################################################################################################

def write_lut_vegetation_indices(lut_path, vi_file_path, band_wavelengths, band_labels, config):
    veg_ind = compute_lut_vegetation_indices(lut_path, band_wavelengths, list(map(str, band_wavelengths)))
    veg_ind_keys = veg_ind.keys()
    with open(vi_file_path, "w") as vi_file:
        vi_file.write(",".join(veg_ind.keys()) + "\n")
        for i in range(len(veg_ind[list(veg_ind_keys)[0]])):
            vi_file.write(",".join([str(veg_ind[key][i]) for key in veg_ind_keys]) + "\n")  # + os.linesep
