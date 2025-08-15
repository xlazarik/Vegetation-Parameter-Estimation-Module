from mosveg.utils import *
import matplotlib as mltp
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from vparam.VPE import vi_approximation_functions
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

mltp.rcParams['agg.path.chunksize'] = 1000


###################################################################################################
# Find optimal parameters of function to fit VI values -> Y-axis (param values = X-axis)
def find_vi_param_fnc_coeffs(function, lut_vi_values, param_values):
    popt, pcov = curve_fit(function, lut_vi_values, param_values, maxfev=15000)
    return popt


def rmse(data, fitter_data):
    return sqrt((1 / len(data)) * np.sum(((data - fitter_data) ** 2)))


def coefficient_of_determination(data, fitted_data):
    # residuals = data - fitted_data
    # 1 - {total sum of squares sum} (yi - y_mean)^2 / {residual sum of squares} (yi - fi)^2
    return 1 - np.sum(np.power(data - np.mean(data), 2)) / np.sum(np.power(data - fitted_data, 2))


# Apply rescaling
def normalize_data(data, from_=0, to_=100, nan_substitute=-1, outlier_margin=0.05):
    # Preprocess
    shape = data.shape
    data = data.flatten()
    data[np.where(np.isneginf(data))] = from_
    data[np.where(np.isinf(data))] = to_

    # Crop outliers
    percentiles = np.nanpercentile(data, [2, 98])
    data[np.where(data < percentiles[0])] = from_
    data[np.where(data > percentiles[1])] = to_

    # Rescale
    scaler = MinMaxScaler((from_, to_))
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return data.reshape(shape[0], shape[1])


# Approximate VP
def approximate_param(function, polynomial_features, normalization, lut_vi_values, image_vi_values, param_values):
    # Exp function which proved produce inaccurate result
    if function == "Interpolate":
        # Train fnc
        fnc_object = interp1d(lut_vi_values, param_values, fill_value="extrapolate")
        # Estimate params
        approximated_values = fnc_object(image_vi_values)
        # Evaluate results
        coeff_of_det = round(coefficient_of_determination(param_values, fnc_object(lut_vi_values)), 3)
        root_mse = round(rmse(param_values, fnc_object(lut_vi_values)), 3)
    elif function == "Linear regression":
        # Train fnc
        co2_X, co2_y = lut_vi_values.values.reshape(-1, 1), param_values
        lin_reg_poly_feat = make_pipeline(
            PolynomialFeatures(degree=polynomial_features, include_bias=False),
            LinearRegression(),
        )
        lin_reg_poly_feat.fit(co2_X, co2_y)
        image_vi_values[np.where(np.isnan(image_vi_values))] = 0
        image_vi_values[np.where(np.isinf(image_vi_values))] = 0
        # Estimate params
        approximated_values = lin_reg_poly_feat.predict(image_vi_values.flatten().reshape(-1, 1))
        approximated_values = approximated_values.reshape(image_vi_values.shape[0], image_vi_values.shape[1])
        space = np.linspace(np.nanmin(lut_vi_values), np.nanmax(lut_vi_values), len(lut_vi_values))
        fitted = lin_reg_poly_feat.predict(space.reshape(-1, 1))
        # Evaluate results
        coeff_of_det = round(coefficient_of_determination(param_values, fitted), 3)
        root_mse = round(rmse(param_values, fitted), 3)
    # Regular empiric function fitting
    else:
        # Fit function to data
        fnc_object = vi_approximation_functions.FUNCTION_MAPPINGS[function]
        fnc_coeffs = find_vi_param_fnc_coeffs(fnc_object, lut_vi_values, param_values)
        # Estimate params
        approximated_values = fnc_object(image_vi_values, *fnc_coeffs)
        # Evaluate results
        coeff_of_det = round(coefficient_of_determination(param_values, fnc_object(lut_vi_values, *fnc_coeffs)), 3)
        root_mse = round(rmse(param_values, fnc_object(lut_vi_values, *fnc_coeffs)), 3)
    if normalization is not None:
        approximated_values = normalize_data(approximated_values, normalization["from"],
                                             normalization["to"], normalization["nans"])

    return approximated_values, coeff_of_det, root_mse


###################################################################################################
# The mappings -> must be matched on the front-end

COLOR_MAPPING = {"Red": "r", "Green": "g", "Blue": "b", "Cyan": "c", "Yellow": "y", "Magenta": "m", "Black": "k",
                 '': "Red"}
SHAPE_MAPPING = {"Point markers": ".", "Pixel markers": ",", "Solid line": "-", "Dashed line": "--",
                 "Dash-dotted line": "-.",
                 "Dotted line": ":", '': "-"}
SCALE_MAPPING = {"Linear": "linear", "Logarithmic": "log", "Symmetrical log": "symlog", "Logit": "logit"}


###################################################################################################

# Preview the fit of the data
# applies LUT filtering, produces a plot, and saves it in the filesystem
def preview_fit(job_id, lut_vi_values, param_values, preview_image_path, veg_index, parameter, preview_settings,
                lut_index_mask):
    if not len(lut_index_mask):
        return ["No records that fulfill required filters."]
    job_db = Job.objects.get(pk=job_id)

    # Plotting -> prepare canvas
    fig = plt.figure()
    space = np.linspace(np.nanmin(lut_vi_values), np.nanmax(lut_vi_values), len(lut_vi_values))
    plt.xscale(SCALE_MAPPING[preview_settings["xscaletype"]])
    plt.yscale(SCALE_MAPPING[preview_settings["yscaletype"]])

    # Apply custom plot configs
    if preview_settings["customscale"] == "true" and preview_settings["xscale"][0] != "" and preview_settings["xscale"][
        1] != "" and preview_settings["xscale"][
        2] != "":
        plt.xticks(np.arange(float(preview_settings["xscale"][0]), float(preview_settings["xscale"][1]),
                             float(preview_settings["xscale"][2])))
        plt.xlim(left=float(preview_settings["xscale"][0]))
        plt.xlim(right=float(preview_settings["xscale"][1]))
        space = np.linspace(float(preview_settings["xscale"][0]), float(preview_settings["xscale"][1]),
                            len(lut_vi_values))
    if preview_settings["customscale"] == "true" and preview_settings["yscale"][0] != "" and preview_settings["yscale"][
        1] != "" and preview_settings["yscale"][
        2] != "":
        plt.yticks(np.arange(float(preview_settings["yscale"][0]), float(preview_settings["yscale"][1]),
                             float(preview_settings["yscale"][2])))
        plt.ylim(bottom=float(preview_settings["yscale"][0]))
        plt.ylim(top=float(preview_settings["yscale"][1]))
    plt.plot(lut_vi_values[lut_index_mask], param_values[lut_index_mask],
             COLOR_MAPPING[preview_settings["bg_colour"]] + SHAPE_MAPPING[preview_settings["bg_type"]])

    info = []
    metrics_info = []
    # Try to plot each curve specified in the request
    for curve in preview_settings["curves"]:
        try:
            # Plot linear reg
            if curve["function"] == "Linear regression":
                # Train
                co2_X, co2_y = lut_vi_values.values.reshape(-1, 1), param_values
                lin_reg_poly_feat = make_pipeline(
                    PolynomialFeatures(degree=curve["poly_features"], include_bias=False),
                    LinearRegression(),
                )
                # Fit
                lin_reg_poly_feat.fit(co2_X, co2_y)
                fitted_data = lin_reg_poly_feat.predict(space.reshape(-1, 1))

            # Empiric fncs
            else:
                # Train
                fnc_object = vi_approximation_functions.FUNCTION_MAPPINGS[curve["function"]]
                popt, pcov = curve_fit(fnc_object, lut_vi_values[lut_index_mask], param_values[lut_index_mask],
                                       check_finite=True,
                                       maxfev=15000 if curve["function"] not in ["Exponential^n", "Logarithm^n",
                                                                                 "Exponential^3",
                                                                                 "Logarithm^3"] else 8000)
                # Fit
                fitted_data = fnc_object(space, *popt)

            # Plot the the function over the input data
            plt.plot(space, fitted_data, COLOR_MAPPING[curve["colour"]] + SHAPE_MAPPING[curve["curve_type"]],
                     linewidth=float(curve["linewidth"]))

            # Compute eval metrics
            coeff_of_det = round(coefficient_of_determination(param_values, fitted_data), 2)
            root_mse = round(rmse(param_values.values, fitted_data), 2)

            metrics_info.append(f"Function: {curve['function']}")
            metrics_info.append(f"COEFFICIENT OF DETERMINATION: {coeff_of_det}")
            metrics_info.append(f"RMSE: {root_mse}")

        except (RuntimeError, ValueError) as e:
            print(e)
            info.append(str(e).split(":")[0])
            update_progress(job_db, "FAILED")

    # Plot legends
    plt.legend(["Lut simulated data"] + [f"{curve['function']}" for curve in preview_settings["curves"]])
    plt.title(f"Vegetation index: {veg_index} - Parameter: {parameter}")
    plt.xlabel(veg_index)
    plt.ylabel(parameter)
    print("Saving preview")
    plt.savefig(preview_image_path)
    plt.close(fig)

    return info, metrics_info
