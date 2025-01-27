import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter

from immstools.dataset import dataset
from immstools.plotting import mobilogram

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import normaltest, shapiro

SQ2PI = 1.0/np.sqrt(2.0*np.pi)


class ATDMultiGauss:

    def __init__(self, centers, **kwargs):

        self._centers = np.array(centers)
        sig = kwargs.get("sigmas", 0.2)  # default width
        try:
            if len(sig) != len(centers):
                raise ValueError("sigmas and centers must have the same length.")
            self._sigmas = sig

        except TypeError:
            # all the same width if a float is provided
            self._sigmas = float(sig)*np.ones_like(self._centers)

        # fit constraints
        self._y0 = kwargs.get("y0", None)  # default is y0 free , otherwise fixed

        self.Xshift = kwargs.get("xshift", False)
        if self.Xshift:
            self.maxshift = kwargs.get("maxshift", np.min(self._sigmas))
        self._xshift = 0.0
        # allow global shift on the x axis, max allowed shift = maxshift

        self.ctol = kwargs.get("ctol", 0.02)  # tolerance on the peak positions
        self.stol = kwargs.get("stol", 0.1)  # tolerance on the peak widths

        self._amplitudes = np.zeros_like(self._centers)
        self._y0 = 0.0

        self._success = False
        self.orig_sigmas = np.copy(self._sigmas)
        self.orig_centers = np.copy(self._centers)
        self._residuals = None

        self._xexp = None
        self._yexp = None

        self.popt = None
        self.perr = None

        self.Rsq = 0.
        self.NTRes = None

    # access to the fit results
    @property
    def ngauss(self):
        return len(self._centers)

    @property
    def success(self):
        return self._success

    @property
    def centers(self):
        return self._centers

    @property
    def sigmas(self):
        return self._sigmas

    @property
    def amplitudes(self):
        return self._amplitudes

    @property
    def y0(self):
        return self._y0

    @property
    def residuals(self):
        return self._residuals

    @property
    def xexp(self):
        return self._xexp

    @property
    def yexp(self):
        return self._yexp

    @property
    def xshift(self):
        return self._xshift

    def fitSummary(self):

        rep = "##############################\n"
        if not(self.success):
            rep += "Fit failed.\n"
            rep += "\n##############################\n"
            return rep
        rep += "Fit with {} Gaussians.\n".format(self.ngauss)

        for i in range(self.ngauss):
            rep += "* peak{}\t@{:.2f} ms, sig={:.2f} ms, A={:.1f}\n".format(
            i,
            self.centers[i],
            self.sigmas[i],
            self.amplitudes[i]
            )
        rep +="baseline y0={:.2f}".format(self.y0)
        if self.Xshift:
            rep +="\nxshift ={:.2f}".format(self.xshift)

        rep+= "\nR² = {:.3f}".format(self.Rsq)
        rep+= "\nNormality of the residuals - p_value = {:.2f}%".format(self.NTRes.pvalue*100.0)

        rep += "\n##############################\n"

        return rep

    # fit function
    def envelope(self, t, *params, **kwargs):

        returnPeaks = kwargs.get("returnPeaks", False)

        params = np.array(params)

        ys = []

        E = np.zeros_like(t)
        y0 = params[-1]
        if self.Xshift:
            xshft =  params[-2]
        else:
            xshft = 0.0

        for i in range(self.ngauss):
            t0 = params[i+self.ngauss] + xshft
            A = params[i]
            sigma = params[i+2*self.ngauss]

            xx = t0 - t
            xx = xx*xx/(2.0*sigma*sigma)

            ys.append((A*SQ2PI/sigma)*np.exp(-xx))

            E = E + ys[-1]

        if returnPeaks:
            ys = np.array(ys)
            return E + y0, ys+y0
        else:
            return E + y0


    # Fitting
    def _reset(self):
        self._amplitudes = np.zeros_like(self._centers)
        self._success = False
        self._sigmas = np.copy(self.orig_sigmas)
        self._centers = np.copy(self.orig_centers)
        self._y0 = 0.0
        self._xshift = 0.0
        self._residuals = None
        self.popt = None
        self.perr = None
        self.Rsq = 0.0
        self.NTRes = None

    def _prepareFit(self, xexp, yexp):

        # 3 params/gaussian+1(+2 if xshift), y0 is at last position, xshift before
        # A0, A1, A2, ... C0, C1, .. sig0, sig1, ..., (xshift), y0.

        if self.Xshift:
            Aguess = np.zeros(self.ngauss*3+2)
            # bounds for the global shift +/- maxshift
            bounds = (np.zeros(self.ngauss*3+2), np.ones(self.ngauss*3+2)*np.inf)
            bounds[0][-2] = -self.maxshift
            bounds[1][-2] = self.maxshift
        else:
            Aguess = np.zeros(self.ngauss*3+1)

            bounds = (np.zeros(self.ngauss*3+1), np.ones(self.ngauss*3+1)*np.inf)

        # bounds for the amplitude >0

        # bounds for the baseline: none
        bounds[0][-1] = -np.inf
        bounds[1][-1] = np.inf

        for c in range(self.ngauss):

            ic = np.where(xexp >= self.centers[c])[0][0]

            # assign the value of the signal as amplitude guess
            Aguess[c] = yexp[ic]/SQ2PI*self.sigmas[c]

            # assign the guess/bounds of the center
            Aguess[c+self.ngauss] = self.centers[c]
            bounds[0][c+self.ngauss] = self.centers[c] - self.ctol
            bounds[1][c+self.ngauss] = self.centers[c] + self.ctol

            # assign the guess/bounds of the width
            Aguess[c+2*self.ngauss] = self.sigmas[c]
            bounds[0][c+2*self.ngauss] = self.sigmas[c] - self.stol
            bounds[1][c+2*self.ngauss] = self.sigmas[c] + self.stol

        return Aguess, bounds

    def fitData(self, D:dataset, **kwargs):

        self._reset()

        # ranges
        mzRange =  kwargs.pop("mzRange", None)
        # default is integrate on full ms
        tdRange =  kwargs.pop("tdRange", None)
        # default is use full ATD range

        # extract profile
        x, y = D.extractIM(mzRange=mzRange, tdRange=tdRange)

        # fit data
        guess, bounds = self._prepareFit(x, y)

        try:
            popt, var_matrix = curve_fit(self.envelope, x, y,
            p0=guess, bounds=bounds)

            perr = np.sqrt(np.diag(var_matrix))

            self._success = True

            popt = np.array(popt)

        except RuntimeError:
            print("WARNING: Unable to fit peak.")
            popt = np.array(guess)
            perr = np.zeros_like(popt)
            return popt, perr

        # store optimized values
        self._amplitudes = popt[0:self.ngauss]
        self._centers = popt[self.ngauss:2*self.ngauss]
        self._sigmas = popt[2*self.ngauss:3*self.ngauss]
        self._y0 = popt[-1]
        if self.Xshift:
            self._xshift = popt[-2]

        # compute residuals
        if self.success:
            self._residuals = self.envelope(x, *popt) - y

        self._xexp = x
        self._yexp = y

        self.popt = popt
        self.perr = perr

        # compute Rsq (as the part of total variance explained)
        ss_res = np.dot(self._residuals,self._residuals)
        ymean = np.mean(self._yexp)
        ss_tot = np.dot((self._yexp-ymean),(self._yexp-ymean))
        # print("RR=", 1-ss_res/ss_tot)
        self.Rsq = 1-ss_res/ss_tot

        if len(self.residuals) > 30:
            self.NTRes = normaltest(self.residuals)
        else:
            self.NTRes = shapiro(self.residuals)

        return popt, perr


    # plotting methods

    def plotFit(self, ax=None, **kwargs):

        showData = kwargs.get("showData", True)
        showPeaks = kwargs.get("showPeaks", True)
        npts = kwargs.get("npts", 200)
        tdRange = kwargs.get("tdRange", [np.min(self.xexp), np.max(self.xexp)])

        t = np.linspace(*tdRange, npts)

        if ax is not None:
            ax = mobilogram(ax, x=self.xexp, y=self.yexp , label='exp.')
        else:
            ax = mobilogram(x=self.xexp, y=self.yexp , label='exp.')

        env, pks = self.envelope(t, *self.popt,returnPeaks=True)
        ax.plot(t, env, 'r-', label = "fit")

        for pk in pks:
            ax.plot(t, pk, 'g--')

        ax.legend()

        return ax

    def plotResiduals(self, **kwargs):
        ax = kwargs.get("ax", None)

        if ax is None:
            fig, ax = plt.subplots()

        ax.tick_params(direction='in', length=5, width=1.5,
                       colors='k', grid_color='k', grid_alpha=0.5)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.set_xlabel("Arrival Time (ms)")
        ax.set_ylabel("Residual")

        ax.axhline(ls='--', c='grey')

        ax.plot(self.xexp, self.residuals, 'ok')
        ymax = 1.1*np.max([np.max(self.residuals), -np.min(self.residuals)])
        ax.set_ylim(-ymax, ymax)


        return ax


class TwoStatesExpDecayModel:

    def __init__(self, **kwargs):
        '''
        Setup an exponential decay model to fit data.

            Optional keyword arguments:
                RO (float): the initial value (at t=0)
                t0 (float): the initial delay (t values will be shifted by t0)
                Rinf (float): the asymptotic value
                postDecay (float): fixed factor to multiply the population
                                    e.g to account for post trapping collision
                                    activation in an IMS/IMS kinetics experiment

            Default is t0=0.0 (no shift) and post decay activation is neglected
            , and R0 and Rinf are free parameters.
        '''

        self.initial_parameters = {
        "R0": None,
        "t0": 0.0,
        "Rinf": None,
        "postDecay": None,
        "tau": None}

        for p in self.initial_parameters:
            self.initial_parameters[p] = kwargs.get(p, None)

        # t0 and postDecay cannot be free
        self.initial_parameters["t0"] = kwargs.get("t0", 0.0)
        self.initial_parameters["postDecay"] = kwargs.get("postDecay", 1.0)

        self.parameters = self.initial_parameters.copy()
        self.parameters_std = self.initial_parameters.copy()
        for pp in self.parameters_std:
            self.parameters_std[pp] = 0.0

        self.popt = None   # the values of the optimized parameters
        self.perr = None   # the stadard errors on the optimized parameters
        self._residuals = None   # the residuals of the fit
        self.Rsq = 0.0   # the R² value of the fit

        # a list of name used to map the values in self.popt
        self._parnames = None

        self._success = False

        self._xexp = None
        self._yexp = None

    # fit results accessors
    @property
    def success(self):
        return self._success

    @property
    def R0(self):
        return self.parameters["R0"]

    @property
    def R0_std(self):
        return self.self.parameters_std["R0"]

    @property
    def Rinf(self):
        return self.parameters["Rinf"]

    @property
    def Rinf_std(self):
        return self.parameters_std["Rinf"]

    @property
    def t0(self):
        return self.parameters["t0"]

    @property
    def postDecay(self):
        return self.parameters["postDecay"]

    @property
    def tau(self):
        return self.parameters["tau"]

    @property
    def tau_std(self):
        return self.parameters_std["tau"]

    # derived parameters

    @property
    def kobs(self):
        return 1000.0/self.parameters["tau"]

    @property
    def kobs_std(self):
        return self.kobs*self.tau_std/self.tau

    @property
    def kforward(self):
        return self.kobs/(1.0 + self.Rinf)

    @property
    def kforward_std(self):
        return self.kforward*(self.kobs_std/self.kobs + self.Rinf_std/self.Rinf)

    @property
    def kbackwards(self):
        return self.kobs - self.kforward

    @property
    def kbackwards_std(self):
        return self.kobs_std + self.kforward_std

    def isFree(self, parameter:str):
        return self.initial_parameters[parameter] is None

    @property
    def residuals(self):
        return self._residuals

    @property
    def xexp(self):
        return self._xexp

    @property
    def yexp(self):
        return self._yexp

    def fitSummary(self):

        rep = "##############################\n"
        if not(self.success):
            rep += "Fit failed.\n"
            rep += "\n##############################\n"
            return rep

        rep += "Fit with 2-states exp. decay.\n"

        if not(self.isFree("R0")):
            rep+="\n Initial ratio fixed to R0={}".format(self.R0)
        if not(self.isFree("Rinf")):
            rep+="\n Asymptotic ratio fixed to Rinf={}".format(self.Rinf)
        if self.t0 != 0:
            rep+="\n assuming initial time shift of t0={}".format(self.t0)
        if self.postDecay != 1.0:
            rep+="\n assuming post decay conversion factor of {}".format(self.postDecay)

        rep+="\n"
        rep+="\n tau = {} (std. err {})".format(self.tau, self.parameters_std["tau"])
        rep+="\n kobs = {} (std. err {})".format(self.kobs, self.kobs_std)

        if self.isFree("R0"):
            rep+="\n Initial ratio R0 = {} (std. err {})".format(self.R0, self.parameters_std["R0"])
        if self.isFree("Rinf"):
            rep+="\n Asymptotic ratio Rinf = {} (std. err {})".format(self.Rinf, self.parameters_std["Rinf"])

        if self.Rinf != 0.0:
            rep+="\n -> kforward = {} (std. err {})".format(self.kforward, self.kforward_std)
            rep+="\n -> kbackwards = {} (std. err {})\n".format(self.kbackwards, self.kbackwards_std)

        rep+= "\nR² = {:.3f}".format(self.Rsq)
        rep+= "\nNormality of the residuals - p_value = {:.2f}%".format(self.NTRes.pvalue*100.0)

        rep += "\n##############################\n"

        return rep

    # fitting function
    def decay(self, x, *par):
        '''
        This is the function used to fit the data.

            Parameters:
                x (numpy array of float): the times at which the function is
                                        evaluated.
                par (numpy array of float): the values of the parameters
                                        of the fit ordered as in _parnames

            Returns:
                an array of values computed for each time in x
        '''

        # parse parameters
        for ii, pnam in enumerate(self._parnames):
            self.parameters[pnam] = par[ii]

        xx = x - self.parameters["t0"]
        R0 = self.parameters["R0"]
        Rinf = self.parameters["Rinf"]
        tau = self.parameters["tau"]
        fac =  self.parameters["postDecay"]

        return fac*(R0*np.exp(-xx/tau) + Rinf*(1.0-np.exp(-xx/tau)))

    # Fitting
    def _reset(self):
        self.parameters = self.initial_parameters.copy()
        for pp in self.parameters_std:
            self.parameters_std[pp] = 0.0
        self.popt = None
        self.perr = None
        self._residuals = None
        self.Rsq = 0.0
        self._success = False

        self._xexp = None
        self._yexp = None

    def _prepareFit(self, x, y):

        self._reset()

        # first guess for tau:
        xmin = np.min(x)
        xmax = np.max(x)

        self._parnames = ["tau"]
        guess = guess = [xmin + 0.2*(xmax-xmin)]

        if self.isFree("R0"):
            self._parnames.append("R0")
            # first guess for R0:
            guess.append(np.max(y))

        if self.isFree("Rinf"):
            self._parnames.append("Rinf")
            # first guess for Rinf:
            guess.append(np.min(y))

        return guess

    def fitData(self, x, y):

        guess = self._prepareFit(x, y)

        try:
            popt, var_matrix = curve_fit(self.decay, x, y,
            p0=guess)

            perr = np.sqrt(np.diag(var_matrix))

            self._success = True

            popt = np.array(popt)

        except RuntimeError:
            print("WARNING: Unable to fit peak.")
            popt = np.array(guess)
            perr = np.zeros_like(coeff)

        # compute residuals
        if self.success:
            self._residuals = self.decay(x, *popt) - y

        self._xexp = x
        self._yexp = y

        self.popt = popt
        self.perr = perr

        # store standard errors
        for ii, pnam in enumerate(self._parnames):
            self.parameters_std[pnam] = self.perr[ii]

        # compute Rsq (as the part of total variance explained)
        ss_res = np.dot(self._residuals,self._residuals)
        ymean = np.mean(self._yexp)
        ss_tot = np.dot((self._yexp-ymean),(self._yexp-ymean))
        # print("RR=", 1-ss_res/ss_tot)
        self.Rsq = 1-ss_res/ss_tot

        # self.NTRes = normaltest(self.residuals)
        if len(self.residuals) > 30:
            self.NTRes = normaltest(self.residuals)
        else:
            self.NTRes = shapiro(self.residuals)

        return self.success

    # plotting methods

    def plotFit(self, ax=None, **kwargs):

        showData = kwargs.get("showData", True)
        npts = kwargs.get("npts", 200)
        tdRange = kwargs.get("tdRange", [np.min(self.xexp), np.max(self.xexp)])

        t = np.linspace(*tdRange, npts)

        if ax is not None:
            ax.plot(self.xexp, self.yexp, 'ko' , label='exp.')
        else:
            fig, ax = plt.subplots()
            ax.plot(self.xexp, self.yexp, 'ko', label='exp.')

        fitcurve = self.decay(t, *self.popt)
        ax.plot(t, fitcurve, 'r-', label = "fit")

        ax.set_xlabel("Trapping Time (ms)")
        ax.set_ylabel("Relative Intensity")

        ax.legend()

        return ax

    def plotResiduals(self, **kwargs):
        ax = kwargs.get("ax", None)

        if ax is None:
            fig, ax = plt.subplots()

        ax.tick_params(direction='in', length=5, width=1.5,
                       colors='k', grid_color='k', grid_alpha=0.5)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.set_xlabel("Trapping Time (ms)")
        ax.set_ylabel("Residual")

        ax.axhline(ls='--', c='grey')

        ax.plot(self.xexp, self.residuals, 'ok')
        ymax = 1.1*np.max([np.max(self.residuals), -np.min(self.residuals)])
        ax.set_ylim(-ymax, ymax)


        return ax