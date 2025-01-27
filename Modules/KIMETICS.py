import pathlib
import csv
import pandas as pd
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.optimize import minimize
from scipy.stats import normaltest, shapiro
from operator import itemgetter
from sklearn.metrics import r2_score
import h5py
import json
import warnings
import toml
from shapely.geometry import LineString
import re
from datetime import datetime
from scipy.signal import find_peaks
import statsmodels.api as sm
from scipy import stats
from lmfit import Parameters, minimize, report_fit, fit_report
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn._oldcore")

R = 8.314
t = 0.650
h = 6.63E-34
kb = 1.38E-23
e = 1.602E-19

def filter_RT(df, RT, dRT) :
    df = df[(abs(df.retention_time.values - RT) < dRT)]
    return df

def filter_MZ(df, MZ, dMZ) :
    df = df[(abs(df.mz.values - MZ) < dMZ)]
    return df

    # df_copy = df.copy()
    # lower_bound = MZ - dMZ
    # upper_bound = MZ + dMZ
    # df_copy.loc[(df_copy['mz'] < lower_bound) | (df_copy['mz'] > upper_bound), 'intensity'] = 0
    # return df_copy

def filter_DT(df, DT, dDT) :
    df_copy = df.copy()
    lower_bound = DT - dDT
    upper_bound = DT + dDT
    df_copy.loc[(df_copy['mz'] < lower_bound) | (df_copy['mz'] > upper_bound), 'intensity'] = 0
    return df_copy

def plot_RT (df, type="scatter", **kwargs) :
    df_copy = df.copy()
    plot = df_copy.groupby("retention_time").sum()
    if type == "scatter" :
        return sns.scatterplot(data=plot, x="retention_time", y="intensity", **kwargs)
    elif type == "line" :
        return sns.lineplot(data=plot, x="retention_time", y="intensity", **kwargs)

def plot_DT (df, type="scatter", **kwargs) :
    df_copy = df.copy()
    plot = df_copy.groupby("drift_time").sum()
    if type == "scatter" :
        return sns.scatterplot(data=plot, x="drift_time", y="intensity", **kwargs)
    elif type == "line" :
        return sns.lineplot(data=plot, x="drift_time", y="intensity", **kwargs)

def plot_MZ (df, type="scatter", **kwargs) :
    df_copy = df.copy()
    plot = df_copy.groupby("mz").sum()
    if type == "scatter" :
        return sns.scatterplot(data=plot, x="mz", y="intensity", **kwargs)
    elif type == "line" :
        return sns.lineplot(data=plot, x="mz", y="intensity", **kwargs)

def extract_DT(df) :
    df_copy = df.copy()
    df_copy = df_copy.groupby("drift_time").sum()
    extracted_df = pd.DataFrame()
    extracted_df["drift_time"] = np.array(df_copy.index)
    extracted_df["intensity"] = df_copy["intensity"].values
    return extracted_df

def extract_RT(df) :
    df_copy = df.copy()
    df_copy = df_copy.groupby("retention_time").sum()
    extracted_df = pd.DataFrame()
    extracted_df["retention_time"] = np.array(df_copy.index)
    extracted_df["intensity"] = df_copy["intensity"].values
    return extracted_df

def extract_MZ(df) :
    df_copy = df.copy()
    df_copy = df_copy.groupby("mz").sum()
    extracted_df = pd.DataFrame()
    extracted_df["mz"] = np.array(df_copy.index)
    extracted_df["intensity"] = df_copy["intensity"].values
    return extracted_df

def sigmoid(x, L , x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

def exponential_decay(x, a, k, c):
    x = np.asarray(x)
    return a * np.exp(-k * x) + c

def two_exponential_decay(x, a_1, k_1, a_2, k_2, c):
    return a_1 * np.exp(-k_1 * x) + a_2 * np.exp(-k_2 * x) + c

def fit_sigmoid(x, y) :
    p0 = [max(y), 10, 1, min(y)]
    bounds = [[0, -np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf, np.inf]]
    popt, pcov = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev = 1000000)
    perr=np.sqrt(np.diag(pcov))
    rep = "##############################"
    rep += "\nSigmoid fit"
    rep += "\nL = {:.2e} \u00B1 {:.2e}".format(popt[0], perr[0])
    rep += "\nx0 = {:.2e} \u00B1 {:.2e}".format(popt[1], perr[1])
    rep += "\nk = {:.2e} \u00B1 {:.2e}".format(popt[2], perr[2])
    rep += "\nb = {:.2e} \u00B1 {:.2e}".format(popt[3], perr[3])
    y_R2 = sigmoid(x, *popt)
    rep += "\nR² = {:.3f}".format(r2_score(y, y_R2))
    rep += "\nSigmoid equation: y = L / (1 + np.exp(-k*(x-x0))) + b"
    rep += "\n##############################"
    return popt, perr, rep, r2_score(y, y_R2)

def fit_exponential_decay(x, y, **kwargs):

    #added on 241118
    x=np.array(x)
    y=np.array(y)
    #

    k_guess = (y.max()-y.min())/(x.max()-x.min())
    initial_guess = [1, k_guess, 0]
    bounds=[[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]]
    y0=kwargs.pop("y0", None)
    if y0!=None:
        bounds[0][2]=y0-1E-5
        bounds[1][2]=y0+1E-5
    popt, pcov = curve_fit(exponential_decay, x, y, p0=initial_guess, bounds=bounds, maxfev = 1000000)
    perr = np.sqrt(np.diag(pcov))
    rep = "##############################"
    rep += "\nTwo-state exponential decay fit"
    rep += "\nA = {:.2e} \u00B1 {:.2e}".format(popt[0], perr[0])
    rep += "\nk = {:.2e} \u00B1 {:.2e}".format(popt[1], perr[1])
    if y0==None:
        rep += "\ny0 = {:.2e} \u00B1 {:.2e}".format(popt[2], perr[2])
    else:
        rep += "\ny0 = {:.2e}".format(y0)
    y_R2 = exponential_decay(x, *popt)
    rep += "\nR² = {:.3f}".format(r2_score(y, y_R2))
    rep += "\nExponential decay Equation: y = A*exp(-k*x) + y0"
    rep += "\n##############################"
    return popt, perr, rep, r2_score(y, y_R2)

def fit_two_exponential_decay(x, y, **kwargs):
    k_guess = (y.max()-y.min())/(x.max()-x.min())
    initial_guess = [0.8, k_guess, 0.2, k_guess, 0]
    bounds = [[0, 0, 0, 0, -1], [2, 1, 2, 1, 1]]
    y0=kwargs.pop("y0", None)
    if y0!=None:
        bounds[0][4]=y0-1E-5
        bounds[1][4]=y0+1E-5
    popt, pcov = curve_fit(two_exponential_decay, x, y, p0=initial_guess, bounds=bounds, maxfev = 1000000)
    perr = np.sqrt(np.diag(pcov))
    rep = "##############################"
    rep += "\nTwo-state exponential decay fit"
    rep += "\nA1 = {:.2e} \u00B1 {:.2e}".format(popt[0], perr[0])
    rep += "\nk1 = {:.2e} \u00B1 {:.2e}".format(popt[1], perr[1])
    rep += "\nA2 = {:.2e} \u00B1 {:.2e}".format(popt[2], perr[2])
    rep += "\nk2 = {:.2e} \u00B1 {:.2e}".format(popt[3], perr[3])
    if y0==None:
        rep += "\ny0 = {:.2e} \u00B1 {:.2e}".format(popt[4], perr[4])
    else:
        rep += "\ny0 = {:.2e}".format(y0)
    y_R2 = two_exponential_decay(x, *popt)
    rep += "\nR² = {:.3f}".format(r2_score(y, y_R2))
    rep += "\nTwo_state exponential decay Equation: y = A1*exp(-k1*x) + A2*exp(-k2*x) + y0"
    rep += "\n##############################"
    return popt, perr, rep, r2_score(y, y_R2)

def MultiGauss(t, *params, **kwargs):

    if len(params)%3 == 1 :
        Xshift = None
    else :
        Xshift = True
    ngauss = int((len(params)-len(params)%3)/3)
    returnPeaks = kwargs.get("returnPeaks", False)

    params = np.array(params)
    ys = []

    E = np.zeros_like(t)
    y0 = params[-1]
    #y0 = 0
    if Xshift is True :
        xshift =  params[-2]
    else:
        xshift = 0.0

    for i in range(ngauss):
        t0 = params[i+ngauss] + xshift
        A = params[i]
        sigma = params[i+2*ngauss]

        xx = t0 - t
        xx = xx*xx/(2.0*sigma*sigma)

        ys.append(A*np.exp(-(t-t0)**2/(2*sigma**2)))

        E = E + ys[-1]

    if returnPeaks:
        ys = np.array(ys)
        return E + y0, ys+y0
    else:
        return E + y0

def fitData(df, separation="DT", **kwargs):

    peaks = kwargs.pop('peaks')
    centers = np.array(peaks)
    ngauss = len(peaks)
    sigmas = kwargs.pop('sigmas')
    sigmas = float(sigmas)*np.ones_like(centers)
    stol = kwargs.pop('stol')
    ctol = kwargs.pop('ctol')
    Xshift = kwargs.pop('Xshift')
    maxshift = kwargs.pop('maxshift')

    orig_sigmas = np.copy(sigmas)
    orig_centers = np.copy(centers)
    amplitudes = np.zeros_like(centers)
    success = False
    sigmas = np.copy(orig_sigmas)
    centers = np.copy(orig_centers)
    y0 = 0.0
    xshift = 0.0
    residuals = None
    popt = None
    perr = None
    Rsq = 0.0
    NTRes = None

    # range
    fit_range = kwargs.pop("fit_range", None)
    if fit_range != None:
        lower_range = fit_range[0]
        upper_range = fit_range[1]

    # extract profile
    if separation == "DT":
        if fit_range:
            df = df[(df["drift_time"] >= lower_range) & (df["drift_time"] <= upper_range)]
        xexp = df["drift_time"].values
        yexp = df["intensity"].values

    elif separation == "RT":
        if fit_range:
            df = df[(df["retention_time"] >= lower_range) & (df["retention_time"] <= upper_range)]
        xexp = df["retention_time"].values
        yexp = df["intensity"].values

    elif separation == "MZ":
        if fit_range:
            df = df[(df["mz"] >= lower_range) & (df["mz"] <= upper_range)]
        xexp = df["mz"].values
        yexp = df["mz"].values
    # fit data

    SQ2PI = 1.0/np.sqrt(2.0*np.pi)
    # 3 params/gaussian+1(+2 if xshift), y0 is at last position, xshift before
    # A0, A1, A2, ... C0, C1, .. sig0, sig1, ..., (xshift), y0.

    if Xshift != None :
        Aguess = np.zeros(ngauss*3+2)
        # bounds for the global shift +/- maxshift
        bounds = (np.zeros(ngauss*3+2), np.ones(ngauss*3+2)*np.inf)
        bounds[0][-2] = -maxshift
        bounds[1][-2] = maxshift
    else:
        Aguess = np.zeros(ngauss*3+1)

        bounds = (np.zeros(ngauss*3+1), np.ones(ngauss*3+1)*np.inf)

    # bounds for the amplitude >0

    # bounds for the baseline: none
    bounds[0][-1] = 0
    bounds[1][-1] = np.inf
    baseline = kwargs.pop('baseline', None)
    if baseline != None:
        bounds[0][-1] = baseline - 1E-5
        bounds[1][-1] = baseline + 1E-5

    for c in range(ngauss):

        ic = np.where(xexp >= centers[c])[0][0]

        # assign the value of the signal as amplitude guess
        Aguess[c] = yexp[ic]

        # assign the guess/bounds of the center
        Aguess[c+ngauss] = centers[c]
        bounds[0][c+ngauss] = centers[c] - ctol
        bounds[1][c+ngauss] = centers[c] + ctol

        # assign the guess/bounds of the width
        Aguess[c+2*ngauss] = sigmas[c]
        bounds[0][c+2*ngauss] = sigmas[c] - stol
        bounds[1][c+2*ngauss] = sigmas[c] + stol

        #test new guess for A
        x_range_lower = centers[c] - ctol
        x_range_upper = centers[c] + ctol
        mask = (xexp >= x_range_lower) & (xexp <= x_range_upper)
        xexp_filtered = xexp[mask]
        yexp_filtered = yexp[mask]
        max_index = np.argmax(yexp_filtered)
        max_yexp = yexp_filtered[max_index]
        Aguess[c] = max_yexp

        #test new guess for centers
        Aguess[c+ngauss] = xexp_filtered[max_index]

    try:
        #popt, var_matrix = curve_fit(MultiGauss, xexp, yexp, p0=Aguess, bounds=bounds, maxfev=10000000)
        #popt, var_matrix = curve_fit(MultiGauss, xexp, yexp, p0=Aguess)
        popt, var_matrix = curve_fit(MultiGauss, xexp, yexp, p0=Aguess, bounds=bounds, method="dogbox")

        perr = np.sqrt(np.diag(var_matrix))

        success = True

        popt = np.array(popt)

    except RuntimeError:
        print("WARNING: Unable to fit peak.")
        popt = np.array(Aguess)
        perr = np.zeros_like(popt)
        rep = "Unable to fit peak."
        return popt, perr, rep

    # store optimized values
    amplitudes = popt[0:ngauss]
    centers = popt[ngauss:2*ngauss]
    sigmas = popt[2*ngauss:3*ngauss]
    y0 = popt[-1]
    if Xshift != None :
        xshift = popt[-2]

    # compute residuals
    if success == True:
        residuals = MultiGauss(xexp, *popt) - yexp

    # compute Rsq (as the part of total variance explained)
    ss_res = np.dot(residuals,residuals)
    ymean = np.mean(yexp)
    ss_tot = np.dot((yexp-ymean),(yexp-ymean))
    # print("RR=", 1-ss_res/ss_tot)
    Rsq = 1-ss_res/ss_tot

    if len(residuals) > 30:
        NTRes = normaltest(residuals)
    else:
        NTRes = shapiro(residuals)
    rep = "##############################\n"
    if success != True:
        rep += "Fit failed.\n"
        rep += "\n##############################\n"
        return rep
    rep += "Fit with {} Gaussians.\n".format(ngauss)

    for i in range(ngauss):
        rep += "* peak{}\t@{:.2f} ms, sig={:.2f} ms, A={:.1f}\n".format(
        i,
        centers[i],
        sigmas[i],
        amplitudes[i]
        )
    rep +="baseline y0={:.2f}".format(y0)

    if Xshift != None :
        rep +="\nxshift ={:.2f}".format(xshift)

    rep+= "\nR² = {:.3f}".format(Rsq)
    rep+= "\nNormality of the residuals - p_value = {:.2f}%".format(NTRes.pvalue*100.0)

    rep += "\n##############################\n"

    return popt, perr, rep

def extract_time(DIR, FILE):
    with open(DIR/pathlib.Path(f'{FILE}/_HEADER.TXT')) as para :
        lines = list(csv.reader(para))
        for line in lines:
            pattern_date = r"\$\$ Acquired Date: (\d{2}-[A-Za-z]{3}-\d{4})"
            match_date = re.search(pattern_date, line[0])
            if match_date:
                date = match_date.group(1)
            else:
                None
            pattern_time = r"\$\$ Acquired Time: (\d{2}:\d{2}:\d{2})"
            match_time = re.search(pattern_time, line[0])
            if match_time:
                time = match_time.group(1)
            else:
                None
    return date + ' ' + time

def calculate_time_difference(datetime_str1, datetime_str2):
    # Define the date format
    date_format = "%d-%b-%Y %H:%M:%S"

    # Parse the datetime strings into datetime objects
    dt1 = datetime.strptime(datetime_str1, date_format)
    dt2 = datetime.strptime(datetime_str2, date_format)

    # Calculate the difference between the two datetime objects
    time_difference = dt2 - dt1

    # Get the difference in seconds
    difference_in_seconds = time_difference.total_seconds()

    return difference_in_seconds

def sec_to_hours(x):
        return x / 3600

def natural_sort_key(s):
    # Split the string into parts (numbers and text)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def keylist(dict) :
    keylist = list(dict[list(dict.keys())[0]].keys())
    return keylist

def line_fit(x, m, p):
    return m*x+p

def MultiGauss_Tcal(params, i, x, ndata, Xshift, **kwargs):

    ys = []
    E = np.zeros_like(x)
    y0 = params[f'y0']

    if Xshift==True:
        ngauss = int((len(params)-1)/(ndata*4))
        for c in range(ngauss):
            t0 = params[f'cen_{i}_{c}'] + params[f'Xshift_{i}_{c}']
            # t0 = params[f'cen_{i}_{c}']
            A = params[f'amp_{i}_{c}']
            sigma = params[f'sig_{i}_{c}']

            xx = t0 - x
            xx = xx*xx/(2.0*sigma*sigma)

            ys.append(A*np.exp(-(x-t0)**2/(2*sigma**2)))

            E = E + ys[-1]
    else:
        ngauss = int((len(params)-1)/(ndata*3))
        for c in range(ngauss):
            t0 = params[f'cen_{i}_{c}']
            A = params[f'amp_{i}_{c}']
            sigma = params[f'sig_{i}_{c}']

            xx = t0 - x
            xx = xx*xx/(2.0*sigma*sigma)

            ys.append(A*np.exp(-(x-t0)**2/(2*sigma**2)))

            E = E + ys[-1]

    returnPeaks = kwargs.get("returnPeaks", False)
    if returnPeaks:
        ys = np.array(ys)
        return E + y0, ys+y0
    else:
        return E + y0

def objective(params, data, Xshift):
    """Calculate total residual for fits of Gaussians to several data sets."""
    new_data=data[:]
    x = new_data.pop(0)
    new_data=np.array(new_data)
    ndata, _ = new_data.shape
    resid = 0.0*new_data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = new_data[i, :] - MultiGauss_Tcal(params, i, x, ndata, Xshift)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def fitData_Tcal(df, **kwargs):

    peaks = kwargs.pop('peaks')
    centers = np.array(peaks)
    ngauss = len(peaks)
    sigma = kwargs.pop('sigma')
    stol = kwargs.pop('stol')
    ctol = kwargs.pop('ctol')
    Xshift = kwargs.pop('Xshift')
    maxshift = kwargs.pop('maxshift')
    baseline = kwargs.pop('baseline')
    orig_sigmas = np.copy(sigma)
    orig_centers = np.copy(centers)
    amplitudes = np.zeros_like(centers)
    success = False
    sigmas = np.copy(orig_sigmas)
    centers = np.copy(orig_centers)
    y0 = 0.0
    xshift = 0.0
    residuals = None
    popt = None
    perr = None
    Rsq = 0.0
    NTRes = None
    SQ2PI = 1.0/np.sqrt(2.0*np.pi)
    # range
    fit_range = kwargs.pop("fit_range", None)
    if fit_range != None:
        lower_range = fit_range[0]
        upper_range = fit_range[1]

    fit_params=Parameters()

    df_keys=[]
    for i in df:
        df_keys.append(i)
    xexp=df[df_keys.pop(0)].to_numpy()

    for iy, y in enumerate(df_keys):

        yexp = df[y].to_numpy()

        # if Xshift == True :
        #     fit_params.add(f'Xshift', value=0.0, min=-maxshift, max=maxshift)

        if baseline != None:
            fit_params.add(f'y0', value=baseline, min=baseline-1E-5, max=baseline+1E-5)
        else:
            fit_params.add(f'y0', value=0.0)

        for c in range(ngauss):

            x_range_lower = centers[c] - ctol
            x_range_upper = centers[c] + ctol
            mask = (xexp >= x_range_lower) & (xexp <= x_range_upper)
            xexp_filtered = xexp[mask]
            yexp_filtered = yexp[mask]
            max_index = np.argmax(yexp_filtered)
            max_yexp = yexp_filtered[max_index]

            cen_guess = xexp_filtered[max_index]

            fit_params.add(f'amp_{iy}_{c}', value=max_yexp, vary=True, min=0.0)
            fit_params.add(f'cen_{iy}_{c}', value=cen_guess, vary=True, min=cen_guess-ctol, max=cen_guess+ctol)
            fit_params.add(f'sig_{iy}_{c}', value=sigma, vary=True, min=sigma-stol, max=sigma+stol)

    for iy in range(1, len(df_keys)):
            for c in range(ngauss):
                fit_params[f'cen_{iy}_{c}'].expr = f'cen_0_{c}'

    for iy in range(1, len(df_keys)):
            for c in range(ngauss):
                fit_params[f'sig_{iy}_{c}'].expr = f'sig_0_{c}'

    out = minimize(objective, fit_params, args=([df[i].to_numpy() for i in df.columns], False))
    print(fit_report(out.params, show_correl=False))
    print(out.success)
    print(out.message)

    if Xshift==True:
        fit_params=out.params
        for iy, y in enumerate(df_keys):
            for c in range(ngauss):
                fit_params.add(f'Xshift_{iy}_{c}', value=0.0, vary=True, min=-maxshift, max=maxshift)

        for c in range(ngauss):
                fit_params[f'cen_{0}_{c}'].set(vary=False)
        out = minimize(objective, fit_params, args=([df[i].to_numpy() for i in df.columns], True))

    popt=out.params

    rep=fit_report(out.params, show_correl=False)
    #rep=out.params.pretty_print()

    print(out.success)
    print(out.message)

    return popt, rep

def print_experiments_setup(WDIR, files):
    parameters_dict = {}
    for k, file in enumerate(files):
            with open(WDIR/pathlib.Path(f'{os.path.splitext(file)[0]}.raw')/pathlib.Path('_extern.inf')) as para :
                line = list(csv.reader(para, delimiter="\t"))
                if line[7][0]=="Instrument Configuration:":
                    place_holder=["", ""]
                    line=place_holder+line
                parameters_dict[f'{os.path.splitext(file)[0]}.raw'] = {
                f'{line[30][0]}':float(line[30][-1]),
                f'{line[31][0]}':float(line[31][-1]),
                f'{line[35][0]}':float(line[35][-1]),
                f'{line[32][0]}':float(line[32][-1]),
                f'{line[33][0]}':float(line[33][-1])}
                if line[113][-1] == "ON":
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[114][0]}']=float(line[114][-1])
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[115][0]}']=float(line[115][-1])
                elif line[113][-1] == "OFF":
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[114][0]}']="add default value"
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[115][0]}']="add default value"
                if line[100][-1] == "ON":
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[101][0]}']=float(line[101][-1])
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[102][0]}']=float(line[102][-1])
                elif line[100][-1] == "OFF":
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[101][0]}']=float(313)
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[102][0]}']=float(4)
                parameters_dict[f'{os.path.splitext(file)[0]}.raw'].update({
                f'{line[49][0]}':float(line[49][-1]),
                f'{line[158][0]} pressure (mbar)':float(line[158][-1]),
                f'{line[87][0]}':float(line[87][-1]),
                f'{line[88][0]}':float(line[88][-1]),
                f'{line[89][0]}':float(line[89][-1]),
                f'{line[90][0]}':float(line[90][-1]),
                f'{line[50][0]}':float(line[50][-1]),
                f'{line[159][0]} pressure (mbar)':float(line[159][-1]),
                f'{line[51][0]}':float(line[51][-1]),
                f'{line[160][0]} pressure (mbar)':float(line[160][-1]),
                f'{line[92][0]}':float(line[92][-1]),
                f'{line[93][0]}':float(line[93][-1]),
                f'{line[94][0]}':float(line[94][-1]),
                f'{line[95][0]}':float(line[95][-1]),
                f'{line[96][0]}':float(line[96][-1]),
                f'{line[104][0]}':float(line[104][-1]),
                f'{line[105][0]}':float(line[105][-1]),
                f'{line[47][0]}':float(line[47][-1]),
                f'{line[98][0]}':float(line[98][-1]),
                f'{line[99][0]}':float(line[99][-1]),
                })
                if line[206][0] == "Trap Collision Energy (eV)" :
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[206][0]}']=float(line[206][-1])
                elif line[206][0] != "Trap Collision Energy (eV)" :
                    parameters_dict[f'{os.path.splitext(file)[0]}.raw'][f'{line[45][0]} (eV)']=float(line[45][-1])
    for i, (key, value) in enumerate(parameters_dict[f'{os.path.splitext(files[2])[0]}.raw'].items()):
        if i == 0:
            print("\n\033[1mSource\033[0m")
        elif i == 5:
            print("\n\033[1mStepWave\033[0m")
        elif i == 7:
            print("\n\033[1mTrap\033[0m")
        elif i == 15:
            print("\n\033[1mMobility cell\033[0m")
        elif i == 26:
            print("\n\033[1mTransfer\033[0m")
        elif i == len(parameters_dict[f'{os.path.splitext(files[2])[0]}.raw']) - 1:
            break
        print(f'{key:<{35}} {value}')

def fit_data_iterative(x_list, y_list, df=None, fit_type="line", r2_threshold=0.97):
    x = np.array(x_list)
    y = np.array(y_list)

    removed_points = []  # Keep track of removed points
    current_r2 = 0

    while current_r2 < r2_threshold and len(x) > 2:
        if fit_type == "line":
            # Fit line using OLS
            X = sm.add_constant(x)  # Add a constant term for the intercept
            model = sm.OLS(y, X).fit()
            predictions = model.predict(sm.add_constant(x))
            residuals = np.abs(y - predictions)
            current_r2 = model.rsquared

            # Print the linear equation
            intercept = model.params[0]
            slope = model.params[1]

        elif fit_type == "sigmoid":
            # Fit sigmoid using curve fitting
            (L, x0, k, b), pcov, _, _ = fit_sigmoid(x, y)
            predictions = sigmoid(x, L, x0, k, b)
            residuals = np.abs(y - predictions)
            current_r2 = r2_score(y, predictions)

        # If R² is good enough, stop
        if current_r2 >= r2_threshold:
            break

        # Identify the worst point (largest residual)
        worst_point_index = np.argmax(residuals)

        if df is None:
            removed_points.append(worst_point_index)

        if df is not None:
            original_index = df.index[worst_point_index]  # Get original index
            removed_points.append(original_index)
            df = df.drop(index=original_index)

        # Remove the worst point
        x = np.delete(x, worst_point_index)
        y = np.delete(y, worst_point_index)

    return x, y, removed_points
