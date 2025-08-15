#!/usr/bin/python
import sys
import argparse
import os
import subprocess
from mosveg.models import Job
from mosveg.utils import update_progress

# DEBUG
# C_LUT_TO_IMAGE_RESAMPLER_PATH = r"/home/xlazarik/mosveg/vparam/utils/executable_scripts/c_image"
# SERVER
C_LUT_TO_IMAGE_RESAMPLER_PATH = r"/home/ubuntu/mosveg/vparam/utils/executable_scripts/c_image"


# Process all files (multiple partitions/fragments) when max batch_size processes can run at once
def batch_subproc_executor(job_id, files, image_wavelengths_fwhms, lut_wavelengths, parts_dir_path, result_dir_path,
                           batch_size=5):
    if job_id != '':
        job_db = Job.objects.get(pk=job_id)
        update_progress(job_db, "RESAMPLING LUT TO IMG RESOLUTION")
    current_i = 0
    # current_limit = batch_size
    proc_deq = []
    to_del = []
    start = time.time()
    logfile = open(result_dir_path + os.path.sep + "logfile.log", "w+")
    errfile = open(result_dir_path + os.path.sep + "errfile.log", "w+")

    retcode = 0

    # Loop until all partitions are processed -> process dequeue is empty
    while current_i < len(files) or len(proc_deq):
        while current_i < len(files) and len(proc_deq) < batch_size:
            # Calling C binary
            proc_deq.append(subprocess.Popen(
                [C_LUT_TO_IMAGE_RESAMPLER_PATH,
                 os.path.join(parts_dir_path, files[current_i]),
                 os.path.join(result_dir_path, os.path.basename(files[current_i]).split(".")[0] + "_RESAMP"),
                 str(len(lut_wavelengths)),
                 ",",
                 str(",".join(map(str, lut_wavelengths))),
                 str(len(image_wavelengths_fwhms["wavelengths"])),
                 str(",".join(map(str, image_wavelengths_fwhms["wavelengths"]))),
                 str(",".join(map(str, image_wavelengths_fwhms["fwhms"])))],
                stdout=logfile,
                stderr=errfile))

            current_i += 1

        # Is process done ? Can it be kicked from the dequeue ?
        for proc_i in range(len(proc_deq)):
            proc_deq[proc_i].wait()
            if proc_deq[proc_i].returncode != 0:
                retcode = proc_deq[proc_i].returncode
            to_del.append(proc_i)

        # Get rid of all p. that are done to make room for new ones
        while to_del:
            proc_i = to_del.pop(0)
            del proc_deq[proc_i]
            to_del = list(map(lambda x: x - 1, to_del))
        if not proc_deq:
            print("EMPTIED QUEUE")
    logfile.write(f">>> Database resampled in: {time.time() - start}s")
    print(f">>> Database resampled in: {time.time() - start}s")
    sout = "LOGS: \n" + logfile.read() + "\n ERRORS: \n" + errfile.read()
    return retcode, sout


def resample_lut_partitions(job_id, parts_dir_path, image_wavelengths_fwhms, lut_wavelengths,
                            result_dir_path=os.getcwd(),
                            batch_size=5, session=""):
    # files contained in parts_dir -> LUT is divided into parts that can be processed in parallel
    if not os.path.exists(parts_dir_path):
        sys.exit(1)

    # Gather files
    files = list(filter(lambda x: x.startswith("{}lut_file_part".format(session)), os.listdir(parts_dir_path)))
    files.sort()

    if not files:
        sys.exit(1)

    # RUNNING THE PROCESSES
    sout = "MONITORING"
    print("MONITORING")
    import time

    a = time.time()
    # Run resampling in parallel on parts
    returncode, sout = batch_subproc_executor(job_id, files, image_wavelengths_fwhms, lut_wavelengths, parts_dir_path,
                                              result_dir_path, batch_size)
    sout += sout
    sout += "TIME ELAPSED: " + str(time.time() - a)
    print("TIME ELAPSED: ", time.time() - a)
    return returncode, sout
