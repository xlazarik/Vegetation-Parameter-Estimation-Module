import shutil
import subprocess
import rasterio
import os


# Writes computed values of VIs and VPs to files
def write_mixed_layer_selection(output_mode, image_path, new_image_path, luts_to_layers_map, image_vegetation_indices,
                                layers):
    dataset = rasterio.open(image_path)

    profile = dataset.profile
    profile.update(
        dtype=rasterio.float64,
        count=len(layers),
        driver='GTiff',
        compress='lzw')

    # DIRS and single band files
    if output_mode == 'dirmode':
        if not os.path.exists(new_image_path):
            os.mkdir(new_image_path)

        profile.update(count=1)

        for layer in layers:
            name = layer["param"] + "_func_" + layer["empiric_function"] + '_index_' + layer["veg_i"] if layer["type"] == "veg_param" else 'index_' + layer["veg_i"]
            with rasterio.open(new_image_path + os.path.sep + name.strip() +".tif", 'w',
                               **profile) as dst:
                if layer["type"] == "veg_param":
                    lut_id = layer["lut"]["lut_id"]
                    dst.write(luts_to_layers_map[lut_id]["approximated_parameters"][
                                  (layer["param"], layer["empiric_function"])][0], 1)
                    dst.set_band_description(1, f"Approximated layer of parameter: {layer['param']} "
                                                f"with coefficient of determination "
                                                f"{luts_to_layers_map[lut_id]['approximated_parameters'][(layer['param'], layer['empiric_function'])][1]} "
                                                f"for {layer['empiric_function']} empiric function")
                else:
                    dst.write(image_vegetation_indices[layer['veg_i'].lower()], 1)
                    dst.set_band_description(1, f"Computed layer of vegetation index: {layer['veg_i']}")
        return

    # Multiband file and layers
    with rasterio.open(new_image_path, 'w', **profile) as dst:
        for i, layer in enumerate(layers):
            if layer["type"] == "veg_param":
                lut_id = layer["lut"]["lut_id"]
                dst.write(
                    luts_to_layers_map[lut_id]["approximated_parameters"][(layer["param"],
                                                                           layer["empiric_function"])][0], i + 1)
                dst.set_band_description(i + 1, f"Approximated layer of parameter: {layer['param']} "
                                                f"with coefficient of determination "
                                                f"{luts_to_layers_map[lut_id]['approximated_parameters'][(layer['param'], layer['empiric_function'])][1]} "
                                                f"for {layer['empiric_function']} empiric function")
            else:
                dst.write(image_vegetation_indices[layer['veg_i'].lower()], i + 1)
                dst.set_band_description(i + 1, f"Computed layer of vegetation index: {layer['veg_i']}")


def convertToGeotiff(image_path, hdr_path, new_path):
    with rasterio.open(image_path) as dataset:
        profile = dataset.profile
        profile.update(
            driver='GTiff',
            compress='lzw')
        raster = dataset.read()
        with rasterio.open(new_path, "w", **profile) as new_dataset:
            new_dataset.write(raster)


def merge_files_by_keyword(dir_path, new_filename="lut_merged_RESAMP", starting_keyword="", contains="",
                           ending_keyword=""):
    files = list(filter(lambda x: x.startswith(starting_keyword) and contains in x and x.endswith(ending_keyword),
                        os.listdir(dir_path)))
    files.sort()

    with open(dir_path + os.path.sep + new_filename, 'wb') as wfd:
        for f in files:
            with open(dir_path + os.path.sep + f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
        print("MERGED")


def count_file_lines(filepath):
    p = subprocess.run(["wc", "-l", filepath], capture_output=True, text=True, shell=False)
    return int(p.stdout.split(" ")[0])


def user_to_lut_params(lut, user_param, from_="approximated"):
    user_params = lut.approximated_parameters if from_ == "approximated" else lut.filtering_parameters
    param_index = user_params.split(lut.delimeter).index(user_param)
    return lut.original_parameters.split(lut.delimeter)[param_index]
