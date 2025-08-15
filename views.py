import base64
import traceback
from datetime import datetime
import shutil
import numpy as np
import pandas
import rasterio
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
import vparam
from mosveg.files import UserStorage
from mosveg.utils import *
from vparam.serializers import LutSerializer
from vparam.VICalculator import vegetation_indices
from vparam.VPE import veg_parameter_approx
from vparam.HDR_parser.hdr_wrapped_obj import load_data, parse_from_json
from vparam.models import Lut, ResampledLutCache
from vparam.VPE.veg_parameter_approx import preview_fit
from mosveg.settings import JOBS_TEMP_DIRECTORY
from vparam.LUT_processor import LUT_processor

##################################################################################
# END points to the REST API of VPE
##################################################################################

# DEBUG
# SYSTEM_LUT_DIR = r"/home/xlazarik/mosveg/lut_databases"
# SYSTEM_SENTINEL2_SRF_NAME = r"/home/xlazarik/mosveg/vparam/utils/config_files/S2-SRF.csv"

# SERVER
from vparam.utils.utils import convertToGeotiff, count_file_lines, user_to_lut_params, write_mixed_layer_selection

SYSTEM_LUT_DIR = r"/mnt/mosveg_store/lut_databases"
SYSTEM_SENTINEL2_SRF_NAME = r"/home/ubuntu/mosveg/vparam/utils/config_files/S2-SRF.csv"

##################################################################################
# Sentinel 2 specification

SYSTEM_SENTINEL2_WAVELENGTHS = [443.0, 490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 945.0, 1375.0, 1610.0,
                                2190.0]
SYSTEM_SENTINEL2_WAVELENGTHS_FILE_BOUNDS = [300.0, 2600.0]
SYSTEM_LANDSAT_SRF_NAME = ""
SYSTEM_LANDSAT_WAVELENGTHS = []
##################################################################################

# V index mapping to wvls

vi_wavelengths_map = {'aoki': ["550", "800"], 'datt2': ["850", "710", "680"], 'cri1': ["510", "550"],
                      'cri2': ["510", "700"], 'tcari-osavi': ["700", "670", "550", "800"],
                      'lcai': ["2165", "2205", "2330"], 'dci': ["705", "722"], 'sr': ["1705", "2155"]}


##################################################################################

# Get HDR metadata
class HdrView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        storage = UserStorage(request.user.uid)
        hdr_path = storage.working_path + os.path.sep + request.query_params['hdr']
        image_path = storage.working_path + os.path.sep + request.query_params['image']

        # SENTINEL 2 -> simply parses tags of the tagged multiband file -> support for tagging required)
        if request.query_params['hdr'] == "NO_HDR":
            import re
            with rasterio.open(image_path) as image:
                hdr_data = {'unit': "", 'number_of_bands': image.count,
                            'wavelengths': [], 'fwhms': [],
                            'reflectance_scale': ""}
                wavelength_regex = "WAVELENGTH:>([0-9.]+)<"
                fwhm_regex = ".*FWHM:>([0-9.]+)<"

                # For each IMG band -> parse from tag
                for i in range(1, image.count + 1):
                    if re.match(wavelength_regex, image.descriptions[i - 1]) and \
                            re.match(fwhm_regex, image.descriptions[i - 1]) and i < len(image.descriptions) + 1:
                        hdr_data["wavelengths"].append(float(re.match(wavelength_regex, image.descriptions[i - 1])[1]))
                        hdr_data["fwhms"].append(float(re.match(fwhm_regex, image.descriptions[i - 1])[1]))
                    else:
                        hdr_data["wavelengths"] = []
                        hdr_data["fwhms"] = []
                        break
            return Response(hdr_data, status.HTTP_200_OK)

        # Here the data are parsed
        hdr_data = load_data(hdr_path)

        return Response({'unit': hdr_data.wwl_units, 'number_of_bands': len(hdr_data.wwl_vector),
                         'wavelengths': hdr_data.wwl_vector, 'fwhms': hdr_data.fwhm_vector,
                         'reflectance_scale': hdr_data.reflectance_scale}, status.HTTP_200_OK)


##################################################################################

# Conversion of hyperspectral data sets to GTIFFs
class FiletypeConvert(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    # Convert (BSQ, BIL, BIP) image to GTIFF
    def post(self, request):
        user_path = UserStorage(request.user.uid).working_path
        # Determine image and HDR file
        try:
            if os.path.splitext(request.data["files"][0])[1] == ".hdr":
                hdr_path = request.data["files"][0]
                image_path = request.data["files"][1]
            else:
                hdr_path = request.data["files"][1]
                image_path = request.data["files"][0]

            new_path = os.path.splitext(user_path + image_path)[0] + ".tif"
            convertToGeotiff(user_path + image_path, user_path + hdr_path, new_path)
        except Exception:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        return Response({}, status=status.HTTP_200_OK)


##################################################################################

# VPE related utility functions
class VparamUtil(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    # Enumerate CSV file (LUT)
    def get(self, request):
        storage = UserStorage(request.user.uid)
        param_path = dict(request.query_params)['param_name'][0]
        return Response(
            {"count_of_lines": count_file_lines(storage.working_path + os.path.sep + param_path)},
            status=status.HTTP_200_OK)

    # Get LUT filtering params (for preview purposes)
    def post(self, request):
        lut = Lut.objects.get(lut_id=int(request.data["lut_id"]))
        param_file = pandas.read_csv(lut.lut_dir + os.path.sep + lut.param_filename,
                                     sep=lut.delimeter, dtype=float, memory_map=True)
        filters = {"": ["", ""]}
        for column in lut.filtering_parameters.split(","):
            if column != "xxx":
                file_column = user_to_lut_params(lut, column, from_="filtered")
                filters[column] = [param_file[file_column].min(), param_file[file_column].max()]
        return Response(filters, status=status.HTTP_200_OK)


##################################################################################

# Compute mask for the filters
def compute_lut_index_mask(simulated_parameter_values, preview, lut):
    index_mask = np.array(range(0, len(simulated_parameter_values[simulated_parameter_values.columns[0]])))
    # Process individual filters
    for obj in preview['lut_filters']:
        if obj["param"] == "":
            continue
        arr = simulated_parameter_values[user_to_lut_params(lut, obj["param"], "filtered")]
        # Default values
        if obj['from'] == "":
            obj['from'] = arr.min() - 0.001
        if obj['to'] == "":
            obj['to'] = arr.max() + 0.001
        obj['from'] = float(obj['from']) - 0.001
        obj['to'] = float(obj['to']) + 0.001
        # Intersection of all sets -> final mask
        restricted_indices = np.where(np.logical_and(arr >= obj['from'], arr <= obj['to']))[0]
        index_mask = np.intersect1d(index_mask, restricted_indices)
    return index_mask


##################################################################################

# Plot previews
class PreviewImage(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    # Make a preview plot
    def post(self, request):
        user_path = UserStorage(request.user.uid).working_path
        layer = request.data["layer"]
        hdr_data = parse_from_json(request.data["hdr_data"])
        preview_settings = request.data["preview"]

        job_id = job_registration(request.user,
                                  ("<<<__SYStem|preview_X__>>>_" + datetime.now().strftime("%d_%m_%H_%M")), False)
        job_db = Job.objects.get(pk=job_id)
        start = time.time()
        info = {"id": job_id, "notes": "", "errors": "", "request": request.data, "logs": "", "time": 0,
                "progress_history": ""}
        update_progress(job_db, "STARTED")
        info["progress_history"] += f"{time.time() - start}s: STARTED\n"

        ##################################################################################
        # Gather specification and parse request config

        lut_id = int(layer["lut"]["lut_id"])
        if hdr_data.wwl_units == 'μm':
            hdr_data.wwl_vector = list(map(lambda x: x * 1000, hdr_data.wwl_vector))
            hdr_data.fwhm_vector = list(map(lambda x: x * 1000, hdr_data.fwhm_vector))

        input_image_path = user_path + os.path.sep + request.data["input_file"]

        required_vegetation_indices = [layer['veg_i'].lower()]

        required_band_wavelengths = vi_wavelengths_map[required_vegetation_indices[0]]
        luts_to_layers_map = {}
        luts_to_layers_map[int(layer["lut"]["lut_id"])] = {"layers": [], "veg_indices": [], "parameters": [],
                                                           "required_wavelengths": [],
                                                           "path_to_resampled_lut": None,
                                                           "approximated_parameters": {}, "object": None}

        if layer["lut"] and layer["type"] == "veg_param":
            luts_to_layers_map[lut_id]["layers"].append(int(layer["layer_id"]))
        if layer["veg_i"] != "":
            luts_to_layers_map[lut_id]["veg_indices"] = required_vegetation_indices
        if layer["param"] != "":
            luts_to_layers_map[lut_id]["parameters"].append((layer["param"], layer["empiric_function"]))

        luts_to_layers_map[lut_id]["required_wavelengths"] = required_band_wavelengths
        luts_to_layers_map[lut_id]["object"] = Lut.objects.get(lut_id=lut_id)

        update_progress(job_db, "COMPUTING IMAGE VEGETATION INDICES")
        info["progress_history"] += f"{time.time() - start}s: COMPUTING IMAGE VEGETATION INDICES\n"

        preview_image_path = ''
        error_info = []
        metrics_info = []

        ##################################################################################
        # Is there a resampled cache that could be used ?
        for lut_id in luts_to_layers_map.keys():
            img_wavelengths = []
            for wavelentgh in luts_to_layers_map[lut_id]["required_wavelengths"]:
                closest_index = vegetation_indices.binary_search(hdr_data.wwl_vector, float(wavelentgh))
                img_wavelengths.append(str(hdr_data.wwl_vector[closest_index]))
            for cache in ResampledLutCache.objects.all():
                if cache.base_lut_id == lut_id and (set(img_wavelengths).issubset(
                        set(cache.resampled_to_wavelengths.split(","))) or set(
                    map(lambda x: str(float(x)), img_wavelengths)).issubset(
                    set(cache.resampled_to_wavelengths.split(",")))):
                    luts_to_layers_map[lut_id]["path_to_resampled_lut"] = cache.resampling_dir

                    break

            ##################################################################################
            # if no fragmet -> resample
            if luts_to_layers_map[lut_id]["path_to_resampled_lut"] is None:
                job_obj = LUT_processor.resample_lut_to_aerial.delay(job_id, JOBS_TEMP_DIRECTORY, hdr_data,
                                                                     luts_to_layers_map[lut_id])
                # job_obj = vparam_controller.resample_lut_to_aerial(job_id, JOBS_TEMP_DIRECTORY, hdr_data,
                #                                                         luts_to_layers_map[lut_id])

                while not job_obj.is_finished and not job_obj.is_failed and not job_obj.is_deferred:
                    time.sleep(2)
                if job_obj.is_failed:
                    update_progress(job_db, "FAILED")
                    info["time"] = time.time() - start
                    job_db.result = info
                    job_db.save()
                    return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                resampled_lut_vi_path, returncode, sout, img_wavelengths = job_obj.result
                # resampled_lut_vi_path, returncode, sout, img_wavelengths = job_obj

                info["logs"] += sout
                luts_to_layers_map[lut_id]["path_to_resampled_lut"] = resampled_lut_vi_path

                cache = ResampledLutCache(
                    owner=request.user,
                    base_lut=luts_to_layers_map[lut_id]["object"],
                    resampling_dir=resampled_lut_vi_path,
                    resampled_to_wavelengths=",".join(list(map(str, sorted(img_wavelengths)))),
                    sentinel=False
                )
                cache.save()

            update_progress(job_db, "COMPUTING LUT VEGETATION INDICES")
            info["progress_history"] += f"{time.time() - start}s: COMPUTING LUT VEGETATION INDICES\n"

            ##################################################################################
            # Compute VI for LUT
            if luts_to_layers_map[lut_id]["parameters"]:
                luts_to_layers_map[lut_id]["lut_vegetation_indices"] = \
                    vegetation_indices.compute_lut_veg_index_selection(
                        luts_to_layers_map[lut_id]["object"],
                        luts_to_layers_map[lut_id]["path_to_resampled_lut"], hdr_data,
                        luts_to_layers_map[lut_id]["required_wavelengths"],
                        luts_to_layers_map[lut_id]["veg_indices"],
                        delimeter=luts_to_layers_map[lut_id]["object"].delimeter)
                #
                simulated_parameter_values = pandas.read_csv(
                    luts_to_layers_map[lut_id]["object"].lut_dir + os.path.sep +
                    luts_to_layers_map[lut_id]["object"].param_filename,
                    sep=luts_to_layers_map[lut_id]["object"].delimeter,
                    dtype=float, memory_map=True)

                ##################################################################################
                # Precompute index mask
                lut_index_mask = compute_lut_index_mask(simulated_parameter_values, preview_settings,
                                                        luts_to_layers_map[lut_id]["object"])

                preview_image_path = JOBS_TEMP_DIRECTORY + os.path.sep + str(job_id) + "_" + \
                                     luts_to_layers_map[lut_id]["object"].lut_name + "_" + layer["veg_i"] + "_" + \
                                     layer["param"] + "_" + layer["empiric_function"] + ".png"
                update_progress(job_db, "PREPARING PREVIEW PLOTS")
                info["progress_history"] += f"{time.time() - start}s: PREPARING PREVIEW PLOTS\n"

                ##################################################################################
                # Create preview plot and save in file system
                errors, metrics = preview_fit(job_id,
                                              luts_to_layers_map[lut_id]["lut_vegetation_indices"][
                                                  layer["veg_i"].lower()],
                                              simulated_parameter_values[
                                                  vparam.utils.utils.user_to_lut_params(
                                                      luts_to_layers_map[lut_id]["object"],
                                                      layer["param"], "approximated")],
                                              preview_image_path, layer["veg_i"], layer["param"],
                                              preview_settings, lut_index_mask)
                error_info.extend(errors)
                metrics_info.extend(metrics)

        info["errors"] += "\n".join(error_info)
        file_content = "".encode("UTF-8")
        if not error_info:
            file_b = open(preview_image_path, "rb")
            file_content = file_b.read()
            os.remove(preview_image_path)
            update_progress(job_db, "FINISHED")
            info["progress_history"] += f"{time.time() - start}s: FINISHED\n"
        info["time"] = time.time() - start
        job_db.result = info
        job_db.save()
        return Response({"plot": base64.b64encode(file_content), "info": error_info, "metrics": metrics_info})


##################################################################################

class LutController(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def delete(self, request):
        storage = UserStorage(request.user.uid)

        lut_ids = [i for i in request.data["luts"]]
        luts = Lut.objects.filter(pk__in=lut_ids)

        ##################################################################################
        # Manage LUT objects
        for lut in luts:
            if request.data["mode"] != "delete":
                if request.data["mode"] == "duplicate":
                    try:
                        shutil.copyfileobj(open(lut.lut_dir + os.path.sep + lut.lut_filename, "r"),
                                           open(storage.working_path + os.path.sep + lut.lut_filename, "w"))
                    except Exception:
                        pass
                    try:
                        shutil.copyfileobj(open(lut.lut_dir + os.path.sep + lut.param_filename, "r"),
                                           open(storage.working_path + os.path.sep + lut.param_filename, "w"))
                    except Exception:
                        pass
                    try:
                        shutil.copyfileobj(open(lut.lut_dir + os.path.sep + lut.srf_filename, "r"),
                                           open(storage.working_path + os.path.sep + lut.srf_filename, "w"))
                    except Exception:
                        pass
                if request.data["mode"] == "unmount":
                    try:
                        os.rename(lut.lut_dir + os.path.sep + lut.lut_filename,
                                  storage.working_path + os.path.sep + lut.lut_filename)
                    except Exception:
                        pass
                    try:
                        os.rename(lut.lut_dir + os.path.sep + lut.param_filename,
                                  storage.working_path + os.path.sep + lut.param_filename)
                    except Exception:
                        pass
                    try:
                        os.rename(lut.lut_dir + os.path.sep + lut.srf_filename,
                                  storage.working_path + os.path.sep + lut.srf_filename)
                    except Exception:
                        pass
            try:
                if request.data["mode"] != "duplicate":
                    shutil.rmtree(lut.lut_dir)
            except Exception:
                pass
        if request.data["mode"] != "duplicate":
            luts.delete()
        return Response(status=status.HTTP_200_OK)

    ##################################################################################
    # Get LUT objects
    def get(self, request):
        user_path = UserStorage(request.user.uid)
        lut_files = dict(request.query_params)['filenames[]']
        file_previews = {}
        for name in lut_files:
            with open(user_path.working_path + os.path.sep + name, "r") as file:
                file_previews[name] = "".join([file.readline() for _ in range(6)])

        return Response({"previews": file_previews},
                        status=status.HTTP_200_OK)

    ##################################################################################
    # Mount LUT
    def post(self, request):
        user_path = UserStorage(request.user.uid)
        if "Czechglobe" not in list(map(lambda x: x["name"], request.user.groups.all().values())):
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        config = request.data["config"]
        # Prepare variables

        approx_parameters = ",".join(list(map(lambda x: x.strip(), config["approximatedParams"])))
        filter_parameters = ",".join(list(map(lambda x: x.strip(), config["filteringParams"])))

        # adjust to more options ticked at once
        if "".join(config["systemSrf"]) == "Sentinel-2":
            config["srfFilename"] = SYSTEM_SENTINEL2_SRF_NAME
            config["srfWavelengths"] = SYSTEM_SENTINEL2_WAVELENGTHS
            config["srfHeader"] = "true"
            config["srfIndexed"] = "true"
        elif "".join(config["systemSrf"]) == "Landsat":
            config["srfFilename"] = SYSTEM_LANDSAT_SRF_NAME
            config["srfWavelengths"] = SYSTEM_LANDSAT_WAVELENGTHS

        ##################################################################################
        # Create new LUT obj and save in db
        job_id = job_registration(request.user, config["lutName"], False)
        lut = Lut(
            lut_name=config["lutName"],
            lut_filename=config["lutFilename"],
            param_filename=config["paramFilename"],
            lut_dir=SYSTEM_LUT_DIR + os.path.sep + config["lutName"],
            resampled_to_sentinel=config["resampleToS2"] == "true",
            fragmented=len(config["fragments"]) > 0,
            no_of_lines=int(config["no_of_records"]),
            crop_type=config["cropType"],
            delimeter=config["lutDelimeter"],
            has_header=config["lutHeader"] == "true",
            units=config["lutUnits"],
            simulation_step=int(config["simulation_step"]),
            approximated_parameters=approx_parameters,
            filtering_parameters=filter_parameters,
            original_parameters="empty",
            lut_scaling_factor=float(config["lutScalingFactor"]),
            lut_wavelength_interval=f"{int(config['lutWavelengths']['low'])},{int(config['lutWavelengths']['high'])}",
            srf_filename=os.path.basename(config["srfFilename"]),
            srf_wavelengths=",".join(map(str, config["srfWavelengths"])),
            srf_file_wavelength_interval=f"{int(SYSTEM_SENTINEL2_WAVELENGTHS_FILE_BOUNDS[0])},{int(SYSTEM_SENTINEL2_WAVELENGTHS_FILE_BOUNDS[1])}",
            srf_header=config["srfHeader"] == "true",
            srf_indexed=config["srfIndexed"] == "true",
            author_information=config["author_information"]
        )
        lut.save()

        ##################################################################################
        # Mount + optimizations (crop/split) ...
        job = LUT_processor.lut_uploaded.delay(request.user, job_id, lut.lut_id,
                                               user_path.working_path + config["lutFilename"],
                                               user_path.working_path + config["paramFilename"],
                                               config["srfFilename"],
                                               SYSTEM_LUT_DIR, config)
        return Response(status=status.HTTP_200_OK)


##################################################################################

# Computation of VI or estimation of VPE
class ComputationController(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    # Fetch lut databases for computation
    def get(self, request):
        databases = Lut.objects.all()
        # response = []
        serializer = LutSerializer(databases, many=True)
        for i in range(len(serializer.data)):
            serializer.data[i]["approximated_parameters"] = list(filter(lambda x: x != "xxx",
                                                                        serializer.data[i][
                                                                            "approximated_parameters"].split(
                                                                            serializer.data[i]["delimeter"])))
            serializer.data[i]["filtering_parameters"] = list(filter(lambda x: x != "xxx",
                                                                     serializer.data[i]["filtering_parameters"].split(
                                                                         serializer.data[i]["delimeter"])))

        return Response(serializer.data, status=status.HTTP_200_OK)

    ##################################################################################
    # Submit computation of VIs and estimation of VPs
    def post(self, request):
        user_path = UserStorage(request.user.uid).working_path
        hdr_data = parse_from_json(request.data["hdr_data"])
        output_files = request.data["output_files"]
        output_mode = request.data["output_mode"]

        # Gather specification and parse request config

        if hdr_data.wwl_units == 'μm':
            hdr_data.wwl_vector = list(map(lambda x: x * 1000, hdr_data.wwl_vector))
            hdr_data.fwhm_vector = list(map(lambda x: x * 1000, hdr_data.fwhm_vector))

        input_image_path = user_path + os.path.sep + request.data["input_file"]
        job_id = job_registration(request.user, str(request.data['output_files'][0]['filename'])[:20] + "...", False)
        job_db = Job.objects.get(pk=job_id)
        start = time.time()
        info = {"id": job_id, "notes": "", "errors": "", "request": request.data, "logs": "", "time": 0,
                "progress_history": ""}
        update_progress(job_db, "STARTED")
        info["progress_history"] += f"{time.time() - start}s: STARTED\n"

        luts_to_layers_map = {}

        required_vegetation_indices = []
        required_band_wavelengths = []

        for output_file in output_files:
            for layer in output_file["layers"]:
                if layer["type"] == "veg_index":
                    required_vegetation_indices.append(layer["veg_i"].lower())
                    continue

                luts_to_layers_map[int(layer["lut"]["lut_id"])] = {"layers": [], "veg_indices": [], "parameters": [],
                                                                   "required_wavelengths": [],
                                                                   "path_to_resampled_lut": None,
                                                                   "approximated_parameters": {}, "object": None}
        ##################################################################################
        # Parse needed VIs and corresponding wavelengths

        for lut_id in luts_to_layers_map.keys():
            for file in output_files:
                for layer in file["layers"]:
                    if layer["lut"] and layer["type"] == "veg_param" and layer["lut"]["lut_id"] == lut_id:
                        luts_to_layers_map[lut_id]["layers"].append([int(layer["layer_id"]), layer['lut_filters']])

                        if layer["veg_i"] != "":
                            luts_to_layers_map[lut_id]["veg_indices"].append(layer["veg_i"].lower())
                            required_vegetation_indices.append(layer["veg_i"].lower())
                        if layer["param"] != "":
                            luts_to_layers_map[lut_id]["parameters"].append(
                                (layer["param"], layer["empiric_function"], layer["lut_filters"],
                                 layer["poly_features"], layer["normalize"]))

            luts_to_layers_map[lut_id]["required_wavelengths"] = \
                sorted(list(set(sum([vi_wavelengths_map[index] for index in
                                     luts_to_layers_map[lut_id]["veg_indices"]], []))))
            required_band_wavelengths += luts_to_layers_map[lut_id]["required_wavelengths"]
            luts_to_layers_map[lut_id]["object"] = Lut.objects.get(lut_id=lut_id)

        required_band_wavelengths = sorted(list(set(sum([vi_wavelengths_map[index] for index in
                                                         required_vegetation_indices], []))))

        update_progress(job_db, "COMPUTING IMAGE VEGETATION INDICES")
        info["progress_history"] += f"{time.time() - start}s: COMPUTING IMAGE VEGETATION INDICES\n"

        ##################################################################################
        # Compute IMG VIs
        image_vegetation_indices, img_vi_error_info = vegetation_indices.compute_image_veg_index_selection(
            input_image_path, hdr_data,
            sorted(list(
                set(required_band_wavelengths))),
            required_vegetation_indices)
        if img_vi_error_info:
            update_progress(job_db, "FAILED")
            info["progress_history"] += f"{time.time() - start}s: FAILED\n"

            info["errors"] += img_vi_error_info
            info["time"] = time.time() - start
            job_db.result = info
            job_db.save()
            return Response(img_vi_error_info, status=status.HTTP_206_PARTIAL_CONTENT)

        # For each LUT do for what it is needed (if various VIs and VPs use LUT,
        # do all operations for LUT and continue with another one
        for lut_id in luts_to_layers_map.keys():
            try:
                img_wavelengths = []
                for wavelentgh in luts_to_layers_map[lut_id]["required_wavelengths"]:
                    closest_index = vegetation_indices.binary_search(hdr_data.wwl_vector, float(wavelentgh))
                    img_wavelengths.append(str(hdr_data.wwl_vector[closest_index]))
                for cache in ResampledLutCache.objects.all():
                    if cache.base_lut_id == lut_id and (set(img_wavelengths).issubset(
                            set(cache.resampled_to_wavelengths.split(","))) or set(
                        map(lambda x: str(float(x)), img_wavelengths)).issubset(
                        set(cache.resampled_to_wavelengths.split(",")))):
                        luts_to_layers_map[lut_id]["path_to_resampled_lut"] = cache.resampling_dir
                        break

                ##################################################################################
                # If no cache -> resample
                if luts_to_layers_map[lut_id]["path_to_resampled_lut"] is None:
                    info["progress_history"] += f"{time.time() - start}s: RESAMPLING LUT STARTED\n"
                    job_obj = LUT_processor.resample_lut_to_aerial.delay(job_id, JOBS_TEMP_DIRECTORY, hdr_data,
                                                                         luts_to_layers_map[lut_id])
                    # job_obj = vparam_controller.resample_lut_to_aerial(job_id, JOBS_TEMP_DIRECTORY, hdr_data,
                    #                                                         luts_to_layers_map[lut_id])

                    while not job_obj.is_finished and not job_obj.is_failed and not job_obj.is_deferred:
                        time.sleep(2)
                    if job_obj.is_failed:
                        update_progress(job_db, "FAILED")
                        info["progress_history"] += f"{time.time() - start}s: FAILED\n"
                        info["time"] = time.time() - start
                        job_db.result = info
                        job_db.save()
                        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                    resampled_lut_vi_path, returncode, sout, img_wavelengths = job_obj.result
                    # resampled_lut_vi_path, returncode, sout, img_wavelengths = job_obj

                    luts_to_layers_map[lut_id]["path_to_resampled_lut"] = resampled_lut_vi_path
                    info["logs"] += sout

                    cache = ResampledLutCache(
                        owner=request.user,
                        base_lut=luts_to_layers_map[lut_id]["object"],
                        resampling_dir=resampled_lut_vi_path,
                        resampled_to_wavelengths=",".join(list(map(str, sorted(img_wavelengths)))),
                        sentinel=False
                    )
                    cache.save()

                ##################################################################################
                # Compute LUT VIs
                if luts_to_layers_map[lut_id]["parameters"]:
                    luts_to_layers_map[lut_id]["lut_vegetation_indices"] = \
                        vegetation_indices.compute_lut_veg_index_selection(
                            luts_to_layers_map[lut_id]["object"],
                            luts_to_layers_map[lut_id]["path_to_resampled_lut"], hdr_data,
                            luts_to_layers_map[lut_id]["required_wavelengths"],
                            luts_to_layers_map[lut_id]["veg_indices"],
                            delimeter=luts_to_layers_map[lut_id]["object"].delimeter)

                    simulated_parameter_values = pandas.read_csv(
                        luts_to_layers_map[lut_id]["object"].lut_dir + os.path.sep +
                        luts_to_layers_map[lut_id]["object"].param_filename,
                        sep=luts_to_layers_map[lut_id]["object"].delimeter,
                        dtype=float, memory_map=True)

                    for (veg_index, (parameter, empiric_function, filters, poly_features, normalize)) in zip(
                            luts_to_layers_map[lut_id]["veg_indices"],
                            luts_to_layers_map[lut_id]["parameters"]):

                        ##################################################################################
                        # Compute LUT mask by filtering
                        index_mask = compute_lut_index_mask(simulated_parameter_values, {"lut_filters": filters},
                                                            luts_to_layers_map[lut_id]["object"])
                        normalization = None
                        if normalize[0] == 'true':
                            normalization = {"from": normalize[1], "to": normalize[2], "nans": -1}
                        ##################################################################################
                        # Estimate VPs
                        luts_to_layers_map[lut_id]["approximated_parameters"][(parameter, empiric_function)] = \
                            veg_parameter_approx.approximate_param(
                                empiric_function,
                                poly_features,
                                normalization,
                                luts_to_layers_map[lut_id]["lut_vegetation_indices"][veg_index][index_mask],
                                image_vegetation_indices[veg_index],
                                simulated_parameter_values[
                                    vparam.utils.utils.user_to_lut_params(luts_to_layers_map[lut_id]["object"],
                                                                          parameter, "approximated")][index_mask])
            # Could not fit the function -> set timeout stopped the search
            except RuntimeError:
                update_progress(job_db, "OPTIMAL PARAMETERS FOR FIT DO NOT EXIST")
                info["progress_history"] += f"{time.time() - start}s: OPTIMAL PARAMETERS FOR FIT DO NOT EXIST\n"

                info["time"] = time.time() - start
                job_db.result = info
                job_db.save()
            # Just catch everything :)
            except Exception as e:
                print(e)
                update_progress(job_db, "FAILED")
                info["progress_history"] += f"{time.time() - start}s: FAILED\n"

                info["errors"] += "Failed with exception : " + str(traceback.format_exc())
                info["time"] = time.time() - start
                job_db.result = info
                job_db.save()
                return Response("", status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # shutil.rmtree(os.path.dirname(luts_to_layers_map[lut_id]["path_to_resampled_lut"]))

        info["progress_history"] += f"{time.time() - start}s: WRITING RESULTS TO OUTPUT FILES\n"
        for j, file in enumerate(output_files):
            output_filename = user_path + file["filename"].replace(" ", "_") + (
                ".tif" if output_mode == 'filemode' else '')
            layers = file["layers"]
            ##################################################################################
            # Now all the necessary values are computed/estimated
            # Write output files
            write_mixed_layer_selection(output_mode, input_image_path, output_filename,
                                        luts_to_layers_map, image_vegetation_indices, layers)

        print("DATAFILES:: ", request.data)
        info["time"] = time.time() - start
        info["progress_history"] += f"{time.time() - start}s: FINISHED\n"

        job_db.result = info
        update_progress(job_db, "FINISHED")
        job_db.save()
        return Response("", status=status.HTTP_200_OK)
