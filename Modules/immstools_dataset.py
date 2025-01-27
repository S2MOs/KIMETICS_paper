# -*- coding: utf-8 -*-
"""
The dataset class is intended to handle IM/MS experimental datasets from .hdf5
files including raw data and metadata. The metadata is used to convert data
channels to m/z and arrival times.

The class allows to:

- extract 2d subsets of the data based on m/z or td ranges

- extract 1d profiles (ATD or mass spec) for selected m/z or td ranges

It also provides access to all information on the dataset (drift conditions,
                                                           etc.)

Copyright (C) 2023 Fabien Chirot

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import warnings
import numpy as np
import h5py
import json
import os

# from measure import Measure
from immstools.measure import Measure

from scipy.optimize import curve_fit

# some constants
kB = 1.38064852E-23  # Boltzmann's constant
pcv = 101325.0 / 760.0  # conversion Torr -> Pa
e0 = 1.6021766208E-19  # elementary charge
Da = 1.660538921E-27  # Da -> kg

sqDa_kBe0 = np.sqrt(Da / kB) / e0

# Mason Schamp's prefactor: we work with ms and yield Ang²
MasonSchampFac = 16.0 / 3.0 * sqDa_kBe0 * 1e-17

def fitParaAsDict(pars):

    pars = np.array(pars)
    # last value is y0
    dd = {'y0': pars[-1], 'peaks':[]}

    # other parameters correspond to different peaks in the order A, mu, w
    # so the remaining parameters shoud be a multiple of 3
    if pars[:-1].shape[0]%3 != 0:
        raise ValueError("parameter list has not the right number of elements: should be 3n+1")
    npeaks = int(pars[:-1].shape[0]/3)
    print(npeaks)

    for i in range(npeaks):
        pkdict = {}
        pkdict['A'] = pars[3*i]
        pkdict['x0'] = pars[3*i+1]
        pkdict['w'] = pars[3*i+2]

        dd['peaks'].append(pkdict)

    return dd


class dataset:
    """
    Handle for an IM/MS dataset stored in a .hdf5 file.

    Attributes
    ----------
    p : float
        Drift pressure (Torr).
    dp : float
        Drift pressure (Torr) standard deviation.
    T : float
        Drift Temperature (K).
    V : float
        Drift voltage (V).
    Ttrap : float
        Trap Temperature (°C)
    z : int
        Ion charge state
    q : float
        Ion charge (C)
    m: float
        Ion mass (Da)

    Methods
    -------
    load(filename:str)
        Loads data from a file.
    extractMS(**kwargs)
        returns two arrays: x the mz scale, and y the intensity
    extractIM(**kwargs)
        returns two arrays: x the td scale, and y the intensity
    subset(**kwargs)
        returns an array corresponding to a subset of the data depending on
        the arguments. returned values are X, Y, Z. X  = td scale (1d),
        Y = mz scale (1d), Z = intensity (2d).
    setupConditions(param: dict)
        Allows to specify parameters not included in the file, especially
        the ion charge and mass. Also allows overriding parameters.
    """

    def __init__(self, filepath=None):

        self.p = None
        self.dp = None

        self.V = None
        self.T = None
        self.L = None

        self.Ttrap = None

        self.z = None
        self.q = None
        self.m = None
        self.mgas = None

        self.sig = None
        self.RESOL = None

        self.chan2mz = np.vectorize(self.__chan2mz)
        self.chan2td = np.vectorize(self.__chan2td)
        self.mz2chan = np.vectorize(self.__mz2chan)
        self.td2chan = np.vectorize(self.__td2chan)

        self._DATA = {}
        self.metadata = {}
        self.settings = {}
        self.datasets = {}

        self.filepath = None
        self.name = None
        self.directory = None

        self.ms = None
        self.im = None

        self.im_x = None
        self.mx_x = None

        self.load(filepath)

        self._Nb_multiFick_peaks = 1

    # load dataset from file

    def _extractMetadata(self, h5Dataset):

        self.metadata = json.loads(h5Dataset.attrs['metadata'])
        self.settings = json.loads(h5Dataset.attrs['settings'])

        # multiple datasets may be contained in one file specified in
        # 'datasets' attr associated with an index
        # default is one single dataset referred as 'data', index -1
        if 'datasets' in h5Dataset.attrs:
            self.datasets = json.loads(h5Dataset.attrs['datasets'])
        else:
            self.datasets = {'data': -1}

        # get acquis. parameters to compute arrival times and m/z
        self.nbTofPts = self.settings["Acquisition"]["main"]['Nb. TOF pts.']
        self.nbImsPts = self.settings["Acquisition"]["main"]['Nb. IMS pts.']

        self.tof_dwell = self.settings["Acquisition"]["main"]['TOF dwell (ns)']
        self.tof_del = self.settings["Acquisition"]["main"]['TOF delay(chan)']

        self.nb_MSSpec_per_point = self.settings["Acquisition"]["main"]["Nb. Mass Spec./pt."]

        # mass calibration
        self.mz1 = self.settings["Acquisition"]["main"]['m/z 1']
        self.mz2 = self.settings["Acquisition"]["main"]['m/z 2']
        self.t1 = self.settings["Acquisition"]["main"]['t 1']
        self.t2 = self.settings["Acquisition"]["main"]['t 2']

        self.tof_mz_min = self.settings["Acquisition"]["main"]['m/z min']
        self.tof_mz_max = self.settings["Acquisition"]["main"]['m/z max']

        self.ims_dwell = self.settings["Acquisition"]["main"]['dt (ms)']
        self.ims_dwell = self.ims_dwell * self.nb_MSSpec_per_point
        self.ims_del = self.settings["Acquisition"]["main"]['tmin (ms)']

        self.ims_tmax = self.settings["Acquisition"]["main"]['tmax (ms)']

        # obtain key drift parameters
        self.V1 = Measure(self.metadata["V1"])
        self.V2 = Measure(self.metadata["V2"])
        self.L1 = self.settings.get("L1", 0.79)
        self.L2 = self.settings.get("L2", 0.79)

        self.mode = self.metadata["mode"]  # single tube or dual tube DEPREC
        if self.metadata['Ref Gate'] == "Gate A":
            # two tubes
            self.L = self.L2 + self.L1
            self.Vdrift = self.V2 + self.V1
        else:
            # only tube 2
            self.L = self.L2
            self.Vdrift = self.V2

        self.Tdrift = Measure(self.metadata["Tdrift"])
        self.pdrift = Measure(self.metadata["pdrift"])

    def _extractData(self, datasetName):

        if self.mode == "SCOPE":
            # print("SCOPE")
            self._DATA = {"data": np.array(datasetName)}

        else:

            # get calibration data
            self.mz1 = np.sqrt(self.mz1)
            self.mz2 = np.sqrt(self.mz2)

            self._a = (self.mz2 - self.mz1) / (self.t2 - self.t1)

            # get data matrix
            self._DATA = {}
            ib = 0
            for dd in self.datasets:
                ie = self.datasets[dd]
                # print ("* loading dataset {} ({}-{})".format(dd, ib,ie))

                if ie == -1:
                    self._DATA[dd] = np.array(datasetName[ib:])
                else:
                    self._DATA[dd] = np.array(datasetName[ib:ie])

                self._DATA[dd] = np.transpose(self._DATA[dd].reshape(
                    self.nbImsPts, self.nbTofPts))
                ib = ie

        # extract full MS and full IM profiles
        self.full_ms = {}
        self.full_im = {}

        if self.mode == "SCOPE":
            for dd in self._DATA:
                self.ms_x = np.array([0])
                self.full_ms[dd] = np.array([1])  # dummy: no MS if SCOPE mode
                self.im_x = self._DATA[dd][0]
                self.full_im[dd] = self._DATA[dd][1]
        else:
            for dd in self._DATA:
                self.full_ms[dd] = self._DATA[dd].sum(axis=1)
                self.full_im[dd] = self._DATA[dd].sum(axis=0)

                self.ms_x = np.arange(self.full_ms[dd].size, dtype=float)
                self.ms_x = self.chan2mz(self.ms_x)

                self.im_x = np.arange(self.full_im[dd].size)
                self.im_x = self.chan2td(self.im_x)

    def load(self, filepath):

        self._DATA = {}
        self.metadata = {}
        self.settings = {}
        self.datasets = {}

        if os.path.exists(filepath):
            self.filepath = filepath
            self.name = os.path.basename(os.path.splitext(self.filepath)[0])
            self.directory = os.path.dirname(self.filepath)
        else:
            self.filepath = None
            self.name = None
            self.directory = None
            warnings.warn("Unable to open file {}".format(self.filepath))
            return

        with h5py.File(filepath, "r") as f:

            h5Dataset = f["data"]

            self._extractMetadata(h5Dataset)
            self._extractData(h5Dataset)

    # access parameters

    def getParam(self, par):
        if par in self.metadata:
            return self.metadata[par]
        if par in self.settings:
            return self.settings[par]
        return None

    def getParamValue(self, par):

        vv = self.getParam(par)

        if vv is None:
            raise NameError('{} is not a valid parameter'.format(par))

        try:
            val = float(vv)
            return val
        except TypeError:
            if isinstance(vv, Measure):
                return vv.asRoundedFloat()

    def metadataList(self):
        return list(self.metadata.keys())

    def __getitem__(self, name):
        return self.getParamValue(name)

    # set parameters values as dictionary (overrides values read in the file)
    # this allows to set the mass and charge of the ions->compute mobilities,
    # etc.
    def setupConditions(self, **kwargs):

        self.p = kwargs.get("p",
                            self.metadata["pdrift"]["average"]
                            ) * 101325.0 / 760.0  # drift pressure in Pa
        self.dp = kwargs.get("dp",
                             self.metadata["pdrift"]["std"]
                             ) * 101325.0 / 760.0  # drift pressure std dev.

        if self.metadata["mode"] == "single":
            # drift voltage in Volt
            self.V = kwargs.get("V",
                                self.metadata["V2"]["average"])

            self.L = kwargs.get("L", 0.79)
        else:
            self.V = kwargs.get("V", self.metadata["V2"]["average"]
                                + self.metadata["V1"]["average"])
            self.L = kwargs.get("L", 0.79 * 2)

        self.T = kwargs.get("T", self.metadata["Tdrift"]["average"])

        self.Ttrap = kwargs.get("Ttrap",
                                self.metadata["Ttrap"]["average"]) + 273.15

        if not ("z" in kwargs or "m" in kwargs):
            return

        self.z = kwargs.get("z", None)  # The charge state of the ions in e
        self.m = kwargs.get("m", None)  # The mass of the ions in Dalton
        self.mgas = kwargs.get("mgas", 4.0)

        # Convert to SI
        self.m = self.m * Da
        self.mgas = self.mgas * Da
        self.q = self.z * e0

        # Compute the theoretical peak width
        self.sig = np.sqrt(2.0 * kB * self.T / (self.q * self.V))
        self.RESOL = 1.0 / (2. * np.sqrt(2. * np.log(2.)) * self.sig)

    # projected profiles

    def extractMS(self, **kwargs):
        """
        Extracts a MS profile for the td range specified as the 'tdrange'
        keyword argument. Returns a full MS if no range is specified.

        Parameters
        ----------
        kwargs : optional
            see notes

        Returns
        -------
        X, Y
            The data points to plot the mass spectrum as two 1d arrays X and Y.

        Notes
        -----
        Keyword arguments include the following:

        mzRange: array_like, optional
            The mass range to return (default = full).

        tdRange: array_like, optional
            The arrival time range over which integration is done.
        """
        X, Y, Z = self.subset(**kwargs)
        Z = Z.sum(axis=1)

        return Y, Z

    def extractIM(self, **kwargs):
        """
        Extracts an ATD profile for the m/z range specified as the 'mzrange'
        keyword argument. Returns a full ATD if no range is specified.

        Parameters
        ----------
        kwargs : optional
            see notes

        Returns
        -------
        X, Y
            The data points to plot the ATD as two 1d arrays X and Y.

        Notes
        -----
        Keyword arguments include the following:

        mzRange: array_like, optional
            The mass range over which integration is done

        tdRange: array_like, optional
            The arrival time range to return (default = full).
        """
        X, Y, Z = self.subset(**kwargs)
        Y = Z.sum(axis=0)

        return X, Y

    def subset(self, **kwargs):
        """
        Extracts a subset of the dataset for the m/z and the td ranges
        specified as the 'mzRange' and 'tdRange' keyword arguments.
        The td scale is returned as X.
        The mz scale is returned as Y.
        The intensities are in the 2d array Z.

        Parameters
        ----------
        kwargs : optional
            see notes

        Returns
        -------
        X, Y, Z
            The td scale is of the subset returned as X.
            The mz scale of the subset is returned as Y.
            The intensities are in the 2d array Z.

        Notes
        -----
        Keyword arguments include the following:

        mzRange: array_like, optional
            The mass range over which integration is done

        tdRange: array_like, optional
            The arrival time range to return (default = full).

        For dataset in Scope mode, the mz dimension does not exist.
        In this case, the Y returns [1]. For consistency, the Z is
        still a 2d array of the form Z = [[i1, i2, ...]]
        """

        # may specify the name of the dataset
        datasetName = kwargs.pop("data", "data")
        # if more than one are handled
        if not (datasetName in self._DATA):
            raise ValueError(datasetName + " does not exist.")

        Z = self._DATA[datasetName]  # the intensities, rows = mz, columns = im
        nr, nc = Z.shape

        # default = full range
        mzRange = kwargs.pop("mzRange", [np.min(self.ms_x), np.max(self.ms_x)])
        tdRange = kwargs.pop("tdRange", [np.min(self.im_x), np.max(self.im_x)])

        # find im channels from im range
        ca = self.td2chan(tdRange[0])
        cb = self.td2chan(tdRange[1])

        if cb < ca:
            ca, cb = cb, ca

        if ca < 0:
            ca = 0

        if cb >= nc:
            cb = nc - 1

        if cb <= ca:
            return

        cb += 1

        itdMin, itdMax = ca, cb

        # find mz channels from mz range
        ca = self.mz2chan(mzRange[0])
        cb = self.mz2chan(mzRange[1])

        if cb < ca:
            ca, cb = cb, ca

        if ca < 0:
            ca = 0

        if cb >= nr:
            cb = nr - 1

        if cb <= ca:
            raise RuntimeError('Could not compute channels for subset.')

        cb += 1

        imzMin, imzMax = ca, cb

        # work with the sub matrix
        Z = Z[imzMin:imzMax, itdMin:itdMax]

        X = self.im_x[itdMin:itdMax]
        Y = self.ms_x[imzMin:imzMax]

        return X, Y, Z

    # fitting
    def fick(self, x, *par):

        A, mu, w, y0 = par
        sigma = mu*self.sig*w

        return A/(sigma*np.sqrt(np.pi))*np.exp(-(x-mu)**2/(2.*sigma**2)) + y0

    def multiFick(self, x, *pars):

        pars = np.array(pars)

        # last value is y0
        y0 = pars[-1]

        # other parameters correspond to different peaks
        pars = np.split(pars[:-1], self._Nb_multiFick_peaks)

        result = 0
        for par in pars:
            # print("pkp", par)
            result += self.fick(x, *par, y0)

        return result

    def fitCurves(self, x, pars):
        """
            Convinience function for plotting fit models from a multi gaussian
            fits. (see fitIMPeaks).

            Parameters
            ----------
            x : the td range over which the values will be computed
            pars : the parameters as provided by the fitIMPeaks method

            Returns
            -------
            envelope :
                numpy array containing the envelope of the fitting function
                over the range
            peaks :
                array of numpy arrays, each containing one of the Gaussian
                functions composing the
                envelope.
        """
        pars = np.array(pars)

        # last value is y0
        y0 = pars[-1]

        # other parameters correspond to different peaks
        pars = np.split(pars[:-1], self._Nb_multiFick_peaks)

        envelope = 0
        peaks = []
        for par in pars:
            peaks.append(self.fick(x, *par, y0))
            envelope += peaks[-1]

        return envelope, peaks

    def fitIMPeaks(self, peaks, **kwargs):
        """
            Fits the ATD profile for the m/z range specified as the 'mzRange'
            keyword argument, over the td range specified as 'tdRange'.
            Full ranges are used if these arguments are omitted.
            Fits are done using Gaussian functions with a constant offset
            using the multiFick method.

            The peaks argument must provide a series of ranges in which peaks
            will be assumed to lay (ex. [[pk1min, pk1max],[pk2min, pk2max]]).

            Parameters
            ----------
            peaks : a series of ranges (doublets) where peaks are expected
            kwargs : optional
                see notes

            Returns
            -------
            pars, errs
                numpy arrays containing the optimized parameters from the fits
                and the associated errors.
                The pars array is orderes as follows:
                [A1, x1, w1, ... An, xn, wn, y0]
                where Ais are the amplitudes of the gaussians, xi, their
                centers, wi a correction factor to their Fickian width, and y0
                is the global offset.
                The standard errors are ordered accordingly. Standard errors
                are provided by the scipy.curve_fit function.

            Notes
            -----
            Keyword arguments include the following:

            mzRange: array_like, optional
                The mass range over which the ATD to fit is extracted
                (default = full).

            tdRange: array_like, optional
                The arrival time range over which fit is done.
            """
        # get fitting region
        # default = full range
        mzRange = kwargs.pop("mzRange", [np.min(self.ms_x), np.max(self.ms_x)])

        tdRange = kwargs.get("tdRange", None)
        if tdRange in ["full", "all"]:
            Xtofit, Ytofit = self.extractIM(mzRange=mzRange)
        elif tdRange is None:
            # in this case the range is defined by the peaks region
            a = np.min(peaks)
            b = np.max(peaks)
            Xtofit, Ytofit = self.extractIM(mzRange=mzRange, tdRange=[a, b])
        else:
            Xtofit, Ytofit = self.extractIM(mzRange=mzRange, tdRange=tdRange)

        pk = []
        guess = []
        pki = []
        # process peak list
        for peak in peaks:
            # find bounds
            a = np.min(peak)
            b = np.max(peak)
            if a == b:
                raise ValueError("Ill-defined peak list.")
            pk.append([a, b])

            # find indexes
            ia = np.where(self.im_x >= a)[0][0]
            ib = np.where(self.im_x >= b)[0][0]
            pki.append([ia, ib])

            # extract peak region
            Xpk, Ypk = self.extractIM(mzRange=mzRange, tdRange=[a, b])

            # first guess: amplitude, center, width/fick
            imax = np.argmax(Ypk)
            guess = guess + [Ypk[imax], Xpk[imax], 1.0]

        guess.append(np.min(Ytofit))  # this is the common y0

        # guess = [A1, x1, w1, ... An, xn, wn, y0]

        # fitting function
        self._Nb_multiFick_peaks = len(pk)
        fifunc = self.multiFick

        try:
            coeff, var_matrix = curve_fit(fifunc, Xtofit, Ytofit, p0=guess)
            perr = np.sqrt(np.diag(var_matrix))

        except RuntimeError:
            print("WARNING: Unable to fit peak.")
            coeff = np.array(guess)
            perr = np.zeros_like(coeff)

        return coeff, perr

    # channel/time or mz conversion
    def __chan2mz(self, c):
        tof = (c + self.tof_del) * self.tof_dwell

        m = self.mz1 + self._a * (tof - self.t1)

        return m * m

    def __mz2chan(self, m):
        m = np.sqrt(m)
        tof = (m - self.mz1) / self._a + self.t1
        c = int(tof / self.tof_dwell) - self.tof_del

        return c

    def __chan2td(self, c):
        return round(c * self.ims_dwell + self.ims_del, 2)

    def __td2chan(self, t):
        return int((t - self.ims_del) / self.ims_dwell)


# if __name__ == "__main__":
#     d = dataset("OGNNQQNY1255_600V_2023724.hdf5")
#     print(d.Vdrift)
#
#     d = dataset("g13L+2_T67_s1_s25.8_d100.0_500V_20221010.hdf5")
#
#     x, y = d.extractIM(mzRange=[1390, 1395], tdRange=[24, 35])
#     print(x)
#
#     print(d.metadata)
