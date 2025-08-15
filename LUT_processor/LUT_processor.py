import json
import subprocess
import numpy as np
import pandas
import math
from vparam.LUT_processor.partition_resampler import resample_lut_partitions
from vparam.utils.utils import merge_files_by_keyword, count_file_lines
import traceback
import shutil
from mosveg.utils import *
import rasterio
from django_rq import job
from mosveg.models import Job
from vparam.models import Lut, LutFragment, ResampledLutCache
from vparam.VICalculator import vegetation_indices



# DEBUG MACHINE
# SPLITTER_PATH = r"/home/xlazarik/mosveg/vparam/utils/executable_scripts/rs_cropper"
# SENTINEL_RESAMPLER_PATH = r"/home/xlazarik/mosveg/vparam/utils/executable_scripts/rs_sentinel"
# SYSTEM_LUT_DIR = r"/home/xlazarik/mosveg/lut_databases"


# SERVER
SYSTEM_LUT_DIR = r"/mnt/mosveg_store/lut_databases"
SPLITTER_PATH = r"/home/ubuntu/mosveg/vparam/utils/executable_scripts/rs_cropper"
SENTINEL_RESAMPLER_PATH = r"/home/ubuntu/mosveg/vparam/utils/executable_scripts/rs_sentinel"


# FNCS ##################################################################################

# Prepare spectral response function (parse) from the CSV file
def process_sentinel_srf(srf_file_path, lut_starting_wavelength, lut_ending_wavelength, srf_starting_wavelength,
                         has_indices):
    bands = {}

    srf_file = pandas.read_csv(srf_file_path, sep=',', dtype=float, memory_map=True)
    upper_bound = lut_ending_wavelength - lut_starting_wavelength

    for band in srf_file:
        if has_indices and band == srf_file.columns[0]:
            continue
        bands[band] = [None, None, None]
        temp = np.nonzero(srf_file[band].array)[0]
        bands[band][0] = temp + srf_starting_wavelength - lut_starting_wavelength

        non_neg_indices = temp[np.where(np.logical_and(bands[band][0] >= 0, bands[band][0] <= upper_bound))[0]]
        bands[band][0] = list(
            map(int, bands[band][0][np.logical_and(bands[band][0] >= 0, bands[band][0] <= upper_bound)]))
        bands[band][1] = list(map(float, srf_file[band][non_neg_indices]))
        bands[band][2] = float(np.sum(bands[band][1]))
        if bands[band][2] == 0:
            bands[band][2] = 1

    return bands


@job('default', timeout='600')
def resample_lut_to_aerial(job_id, temp_path, hdr_data, lut_entry):

    # Find smallest possible fragment
    fragments = LutFragment.objects.filter(base_lut=lut_entry["object"])
    wavelength_difference = math.inf
    minimal_fragment = None

    for fragment in fragments:
        if fragment.low_wavelength_bound <= float(lut_entry["required_wavelengths"][0]) \
                and fragment.high_wavelength_bound >= float(lut_entry["required_wavelengths"][-1]):
            diff = abs(float(lut_entry["required_wavelengths"][0]) - fragment.low_wavelength_bound + \
                       float(lut_entry["required_wavelengths"][1]) - fragment.high_wavelength_bound)

            if diff < wavelength_difference:
                wavelength_difference = diff
                minimal_fragment = fragment

    image_wavelengths_fwhms = {"wavelengths": [], "fwhms": []}

    # Which wavelengths are closest to VI required wavelengths ? -> LUT will be resampled to those
    for wavelentgh in lut_entry["required_wavelengths"]:
        closest_index = vegetation_indices.binary_search(hdr_data.wwl_vector, float(wavelentgh))
        image_wavelengths_fwhms["wavelengths"].append(
            hdr_data.wwl_vector[closest_index])
        image_wavelengths_fwhms["fwhms"].append(hdr_data.fwhm_vector[closest_index])

    lut_wavelengths = [i for i in range(int(minimal_fragment.low_wavelength_bound),
                                        int(minimal_fragment.high_wavelength_bound) + 1)]

    result_dir_path = temp_path + os.path.sep + str(job_id) + "_" + \
                      lut_entry["object"].lut_name + "_" + minimal_fragment.fragment_name
    try:
        os.mkdir(result_dir_path)
    except OSError:
        print("Creation of the directories %s failed" % result_dir_path)
    else:
        print("Successfully created the directory %s " % result_dir_path)

    # Resample partitions in parallel
    returncode, sout = resample_lut_partitions(job_id,
                                               minimal_fragment.fragment_dir,
                                               image_wavelengths_fwhms,
                                               lut_wavelengths,
                                               result_dir_path,
                                               len(os.listdir(minimal_fragment.fragment_dir)))

    # Create merged file (its small now) from the partitions
    merge_files_by_keyword(result_dir_path, "lut_resampled_merged.csv", "lut_file_part", ending_keyword="RESAMP")
    resampled_lut_vi_path = result_dir_path + os.path.sep + "lut_resampled_merged.csv"
    return resampled_lut_vi_path, returncode, sout, image_wavelengths_fwhms["wavelengths"]


#####################################################################################################

# Run only once when mounted
def resample_lut_to_sentinel2(lut_path, resampler_path, lut_object):
    resampled_path = os.path.dirname(lut_path) + os.path.sep + "sentinel2" + os.path.sep + \
                     os.path.basename(lut_path).split(".")[
                         0] + "_sentinel2_resampled.csv"
    if not os.path.exists(os.path.dirname(lut_path) + os.path.sep + "sentinel2"):
        os.mkdir(os.path.dirname(lut_path) + os.path.sep + "sentinel2")

    logfile = open(os.path.dirname(lut_path) + os.path.sep + "logfile.log", "a+")
    errfile = open(os.path.dirname(lut_path) + os.path.sep + "errfile.log", "a+")

    # Get SRF kernel for record weighted summing
    srf = process_sentinel_srf(lut_object.lut_dir + os.path.sep + lut_object.srf_filename,
                               int(float(lut_object.lut_wavelength_interval.split(",")[0])),
                               int(float(lut_object.lut_wavelength_interval.split(",")[1])),
                               int(float(lut_object.srf_file_wavelength_interval.split(",")[0])),
                               lut_object.srf_indexed)

    # Rust binary, resampling of LUT to S2
    sentinel_resampler = subprocess.Popen([resampler_path, lut_path, resampled_path, json.dumps(srf, indent=4)],
                                          stdout=logfile,
                                          stderr=errfile,
                                          shell=False)
    sentinel_resampler.wait()
    sout = "LOGS: \n" + logfile.read() + "\n ERRORS: \n" + errfile.read()
    return resampled_path, sentinel_resampler.returncode, sout


#####################################################################################################

# Split to partitions that can be processed in parallel -> horizontal reduction
def split_lut(lut_path, param_path, parts_dir_path, splitter_path, config, low, high, partitions_num=5):
    sout = ""
    if not os.path.exists(parts_dir_path):
        os.mkdir(parts_dir_path)
    logfile = open(os.path.dirname(lut_path) + os.path.sep + "logfile.log", "w+")
    errfile = open(os.path.dirname(lut_path) + os.path.sep + "errfile.log", "w+")

    if int(config["no_of_records"]) == 0:
        # We need to enumerate file to know partition size (number of records)
        config["no_of_records"] = count_file_lines(param_path)

    # Rust script splitting the LUT
    lut_splitter = subprocess.Popen(
        [splitter_path, parts_dir_path, lut_path, str(partitions_num), config["lutDelimeter"],
         str(config["simulation_step"]), str(low - int(config["lutWavelengths"]["low"]) - 40),
         str(high - int(config["lutWavelengths"]["low"]) + 40),
         str(config["no_of_records"])],
        stdout=logfile, stderr=errfile, shell=False)
    lut_splitter.wait()
    sout = "LOGS: \n" + logfile.read() + "\n ERRORS: \n" + errfile.read()

    return lut_splitter.returncode, sout


#####################################################################################################


@job('default', timeout='4h')
def lut_uploaded(user, job_id, lut_id, current_lut_path, param_file_path, srf_file_path, system_lut_dir, config):
    job_db = Job.objects.get(pk=job_id)
    update_progress(job_db, "STARTED")
    start = time.time()
    info = {"id": job_id, "notes": "", "errors": "", "request": config, "logs": "", "time": 0,
            "progress_history": ""}
    info["progress_history"] += f"{time.time() - start}s: STARTED\n"

    try:
        # Get ready paths and dirs
        lut = Lut.objects.get(pk=lut_id)
        lut_subdir = config["lutName"]
        new_lut_path = system_lut_dir + os.path.sep + lut_subdir + os.path.sep + config["lutFilename"]
        new_param_path = system_lut_dir + os.path.sep + lut_subdir + os.path.sep + config["paramFilename"]
        new_srf_path = system_lut_dir + os.path.sep + lut_subdir + os.path.sep + os.path.basename(config["srfFilename"])
        os.mkdir(system_lut_dir + os.path.sep + lut_subdir)

        with open(param_file_path, "r") as param_file:
            param_lines = param_file.readlines()
        lut.original_parameters = param_lines[0].strip()
        lut.save()

        os.rename(param_file_path, new_param_path)
        if config["systemSrf"] == "":
            os.rename(srf_file_path, new_srf_path)
        else:
            shutil.copyfile(srf_file_path, new_srf_path)
        if not lut.has_header:
            with open(new_lut_path, 'wb') as wfd:
                header = ",".join([f"W{column}" for column in range(
                    int(float(lut.lut_wavelength_interval.split(",")[0])), int(
                        float(lut.lut_wavelength_interval.split(",")[1])) + 1)])
                wfd.write(str.encode(f"{header}\n"))
                with open(current_lut_path, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                os.remove(current_lut_path)
                print("HEADER PADDED")
                lut.has_header = True
        else:
            os.rename(current_lut_path, new_lut_path)

        # Create individual fragments
        for fragment in config["fragments"]:
            fragment_name = "fragment_" + str(fragment["low"]) + "_" + str(fragment["high"])
            fragment_path = system_lut_dir + os.path.sep + lut_subdir + os.path.sep + fragment_name

            update_progress(job_db, f"CREATING FRAGMENT: {fragment_name}")
            info["progress_history"] += f"{time.time() - start}s: CREATING FRAGMENT: {fragment_name}\n"

            # Make paralelizable
            code, sout = split_lut(new_lut_path, new_param_path, fragment_path, SPLITTER_PATH,
                                   config, fragment["low"], fragment["high"], partitions_num=config["no_of_partitions"])
            if code != 0:
                update_progress(job_db, "FAILED")
                info["progress_history"] += f"{time.time() - start}s: FAILED\n"
                info["errors"] += f"Non-zero retcode: {code}"
                info["logs"] = sout
                info["time"] = time.time() - start
                job_db.result = info
                job_db.save()

                return

            # Save fragments
            lut_fragment = LutFragment(fragment_dir=fragment_path, base_lut=lut,
                                       fragment_name=fragment_name,
                                       low_wavelength_bound=fragment["low"] - 40,
                                       high_wavelength_bound=fragment["high"] + 40)
            lut_fragment.save()

        update_progress(job_db, "RESAMPLING TO SENTINEL")
        info["progress_history"] += f"{time.time() - start}s: RESAMPLING TO SENTINEL\n"

        # SENTINEL 2
        resampled_path, code, sout = resample_lut_to_sentinel2(new_lut_path, SENTINEL_RESAMPLER_PATH, lut)

        band_wavelengths = [443.0, 490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 945.0, 1375.0, 1610.0,
                            2190.0]
        band_labels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13']

        # Create LUT cache for Sentinel2
        cache = ResampledLutCache(
            owner=user,
            base_lut=lut,
            resampling_dir=resampled_path,
            resampled_to_wavelengths=",".join(list(map(str, sorted(band_wavelengths)))),
            sentinel=True
        )
        cache.save()
        vi_filepath = lut_subdir + os.path.sep + "sentinel2" + os.path.sep + \
                      os.path.basename(config["lutName"]) + "_sentinel_resampled_vi_index.csv"

        # Precompute VIs
        vegetation_indices.write_lut_vegetation_indices(resampled_path, SYSTEM_LUT_DIR + os.path.sep + vi_filepath,
                                                        band_wavelengths, band_labels,
                                                        config)

    except Exception as e:
        print(e)
        update_progress(job_db, "FAILED")
        info["errors"] += "Failed with exception : " + str(traceback.format_exc())
        info["progress_history"] += f"{time.time() - start}s: FAILED\n"

    info["time"] = time.time() - start
    info["progress_history"] += f"{time.time() - start}s: FINISHED\n"
    job_db.result = info
    update_progress(job_db, "FINISHED")
    job_db.save()

