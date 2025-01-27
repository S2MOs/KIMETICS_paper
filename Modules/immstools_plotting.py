""" Convenience functions for plotting IMS data using matplotlib

    See Also
    --------
    dataset.dataset

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

import numpy as np
import matplotlib.pyplot as plt

# from dataset import dataset
from immstools.dataset import dataset


# plotting functions (using matplotlib)


def massSpec(ax=None, **kwargs):
    """Plots a mass spectrum using matplotlib

The plot can be done either from a dataset or from two 1d arrays.

    Parameters
    ----------
    ax: matplotlib.Axes, optional
        Existing matplotlib axis object, created if None

    kwargs: optional

    Returns
    -------
    matplotlib.Axes
        The ax created or passed as argument
    Notes
    -----
    Keyword arguments include the following:

    dataset: dataset, optional
            The dataset to be processed.

    x, y: array_like
        The data to plot.

    IMPORTANT: Either 'dataset' or both 'x' and 'y' have to be provided.

    mzRange: array_like, optional
        The mass range to display

    tdRange: array_like, optional
        The arrival time range over which integration is done

    All extra kwargs options are passed to matplotlib axis.plot

    See Also
    --------
    dataset.dataset

    Examples
    --------
    Dummy plot from two data arrays (b plotted as a function of a).

    >>> a = [1, 2, 3]
    >>> b = [3, 4, 5]
    >>> massSpec(x=a, y=b, color='blue')

    """
    d = kwargs.pop("dataset", None)

    if d is not None:
        if not (isinstance(d, dataset)):
            raise ValueError("must provide a dataset.")
        mz, intensity = d.extractMS(**kwargs)
        # avoid to pass range arguments to plot functions
        kwargs.pop("mzRange", None)
        kwargs.pop("tdRange", None)
    else:
        mz = kwargs.pop("x", None)
        intensity = kwargs.pop("y", None)

    figsize = kwargs.pop("figsize", (6, 4))
    layout = kwargs.pop("layout", 'constrained')

    if ax is None:
        # then generate a graph
        fig, ax = plt.subplots(figsize=figsize, layout=layout)

    xlabel = kwargs.pop("xlabel", 'm/z')
    ylabel = kwargs.pop("ylabel", 'Intensity (arb. units)')

    normalize = kwargs.pop("normalize", None)
    if normalize == 'sum':
        intensity = intensity / np.sum(intensity)
    elif normalize == 'max':
        intensity = intensity - np.min(intensity)
        intensity = intensity / np.max(intensity)

    # axis style
    style = kwargs.pop("style", 'boxed')
    if style == 'boxed':
        ax.tick_params(direction='in', length=5, width=1.5, colors='k',
                       grid_color='k', grid_alpha=0.5)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif style == 'noy':
        ax.tick_params(axis='x', direction='out', length=5, width=1.5,
                       colors='k', grid_color='k', grid_alpha=0.5)

        for axis in ['top', 'left', 'right']:
            ax.spines[axis].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_visible(False)

    # plot style
    kwargs["linewidth"] = kwargs.get("linewidth", 1)
    kwargs["color"] = kwargs.get("color", 'black')
    ax.plot(mz, intensity, **kwargs)

    return ax


def mobilogram(ax=None, **kwargs):
    """Plots a mass mobilogram using matplotlib

    The plot can be done either from a dataset or from two 1d arrays.

        Parameters
        ----------
        ax: matplotlib.Axes, optional
            Existing matplotlib axis object, created if None

        kwargs: optional

        Returns
        -------
        matplotlib.Axes
            The ax created or passed as argument
        Notes
        -----
        Keyword arguments include the following:

        dataset: dataset, optional
                The dataset to be processed.

        x, y: array_like
            The data to plot.

        IMPORTANT: Either 'dataset' or both 'x' and 'y' have to be provided.

        mzRange: array_like, optional
            The mass range over which integration is done

        tdRange: array_like, optional
            The arrival time range to display

        All extra kwargs options are passed to matplotlib axis.plot

        See Also
        --------
        dataset.dataset
        """
    d = kwargs.pop("dataset", None)

    if d is not None:
        if not (isinstance(d, dataset)):
            raise ValueError("must provide a dataset.")
        driftTime, intensity = d.extractIM(**kwargs)
        # avoid to pass range arguments to plot functions
        kwargs.pop("mzRange", None)
        kwargs.pop("tdRange", None)
    else:
        driftTime = kwargs.pop("x", None)
        intensity = kwargs.pop("y", None)

    figsize = kwargs.pop("figsize", (6, 4))
    layout = kwargs.pop("layout", 'constrained')

    if ax is None:
        # then generate a graph
        fig, ax = plt.subplots(figsize=figsize, layout=layout)

    xlabel = kwargs.pop("xlabel", 'Arrival Time (ms)')
    ylabel = kwargs.pop("ylabel", 'Intensity (arb. units)')

    normalize = kwargs.pop("normalize", None)
    if normalize == 'sum':
        intensity = intensity / np.sum(intensity)
    elif normalize == 'max':
        intensity = intensity - np.min(intensity)
        intensity = intensity / np.max(intensity)

    # axis style
    style = kwargs.pop("style", 'boxed')
    if style == 'boxed':
        ax.tick_params(direction='in', length=5, width=1.5,
                       colors='k', grid_color='k', grid_alpha=0.5)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif style == 'noy':
        ax.tick_params(axis='x', direction='out', length=5,
                       width=1.5, colors='k', grid_color='k', grid_alpha=0.5)

        for axis in ['top', 'left', 'right']:
            ax.spines[axis].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_visible(False)

    # plot style
    kwargs["marker"] = kwargs.get("marker", 's')
    kwargs["markersize"] = kwargs.get("markersize", 3)
    kwargs["linewidth"] = kwargs.get("linewidth", 1)
    kwargs["color"] = kwargs.get("color", 'black')

    ax.plot(driftTime, intensity, **kwargs)

    return ax


def plotAsWaterfall(datasets, **kwargs):
    """Plots a series of ATDs with common td scale as a waterfall
    from dataset objects


        Parameters
        ----------

        datasets: array_like , optional
                The list of dataset to be processed.

        kwargs: optional

        Returns
        -------
        matplotlib.Axes
            The ax created or passed as argument
        Notes
        -----
        Keyword arguments include the following:

        levels: array_like , optional
            the list of "z" value corresponding to each dataset,
            must be the same size as 'datasets'

        levelParam: string, optionnal
            the name of the dataset parameter to use as the level scale.

        mzRange: array_like, optional
            The mass range for which the ATD is extracted

        tdRange: array_like, optional
            The arrival time range to display

        All extra kwargs options are passed to matplotlib axis.plot

        See Also
        --------
        dataset.dataset
        """

    x = []
    y = []
    param = kwargs.pop("levelParam", None)
    if param is not None:
        levels = []

    normalize = kwargs.pop("normalize", None)
    for d in datasets:
        if not (isinstance(d, dataset)):
            raise ValueError("must provide a list of datasets.")

        if param is not None and param in d.metadata:
            pp = d.getParam(param)
            if isinstance(pp, dict):
                pp = d.getParam(param)['average']
            levels.append(pp)

        # extract data to plot
        driftTime, intensity = d.extractIM(**kwargs)

        # normalize if necessary
        if normalize == 'sum':
            intensity = intensity / np.sum(intensity)
        elif normalize == 'max':
            intensity = intensity - np.min(intensity)
            intensity = intensity / np.max(intensity)

        # store
        x.append(driftTime)
        y.append(intensity)

        # check if x scales match, otherwise shout!
        if len(x) > 1:
            if x[-1][0] != x[-2][0] or \
                x[-1][-1] != x[-2][-1] or \
                x[-1][1] - x[-1][0] != x[-2][1] - x[-2][0] or \
                len(x[-1]) != len(x[-2]):

                raise ValueError("The x axes must be the \
                                 same for all datasets.")

    if param is not None:
        kwargs["levels"] = levels

    return waterfall(x[0], y, **kwargs)


def waterfall(x, y, **kwargs):
    """Plots a series of datasets with common x scale as a waterfall


        Parameters
        ----------

        x: array_like , optional
                The x scale for the plots, common to all.

        y: array_like , optional
                The list of datasets to plot.
                Each dataset must atch the x scale.

        kwargs: optional

        Returns
        -------
        matplotlib.Axes
            The ax created or passed as argument
        Notes
        -----
        Keyword arguments include the following:

        levels: array_like , optional
            the list of "z" value corresponding to each dataset
            must be the same size as 'datasets'

        mzRange: array_like, optional
            The mass range for which the ATD is extracted

        tdRange: array_like, optional
            The arrival time range to display

        All extra kwargs options are passed to matplotlib axis.plot

        See Also
        --------
        dataset.dataset
        """

    XX = np.array(x)
    Z = np.array(y)

    # define the levels scale
    levels = kwargs.pop("levels", np.arange(len(XX)))
    if not (len(Z) == len(levels)):
        raise ValueError("The number of levels must be equal \
                         to the number of datasets.")
    # levels = levels - np.min(levels)
    # YY = levels/np.max(levels)
    YY = levels

    X, Y = np.meshgrid(XX, YY)

    # avoid to pass range arguments to plot functions
    kwargs.pop("mzRange", None)
    kwargs.pop("tdRange", None)

    figsize = kwargs.pop("figsize", (6, 4))
    layout = kwargs.pop("layout", 'constrained')

    # generate a graph
    fig = plt.figure(figsize=figsize, layout=layout)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    showz = kwargs.pop("showz", False)
    zlabel = kwargs.pop("zlabel", 'Intensity (arb. units)')
    xlabel = kwargs.pop("xlabel", 'Arrival Time (ms)')
    ylabel = kwargs.pop("ylabel", '')

    # axis style

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    color = kwargs.pop("tdRange", 'k')

    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0, color=color)
    if showz:
        ax.set_zlabel(zlabel)
    else:
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

    return ax


def map2d(d: dataset, ax=None, **kwargs):
    """Plots a 2d map (greyscale) from an IMS/MS dataset using either
    a contour plot (contourf) a a grid (pcolormesh).
        This behaviour is controlled by the 'style' option
        (to enable contour, use style='contour').

        Parameters
        ----------
        d: dataset
            The dataset to plot.

        ax: matplotlib.Axes, optional
            Existing matplotlib axis object, created if None.

        kwargs: optional

        Returns
        -------
        matplotlib.Axes
            The ax created or passed as argument

        Notes
        -----
        Keyword arguments include the following:

        style: string, optional
                enable contour plot by setting to 'contour'.

        mzRange: array_like, optional
            The mass range to display

        tdRange: array_like, optional
            The arrival time range to display

        All extra kwargs options are passed to matplotlib axis.plot

        See Also
        --------
        dataset.dataset
        """
    X, Y, Z = d.subset(**kwargs)
    # avoid to pass range arguments to plot functions
    kwargs.pop("mzRange", None)
    kwargs.pop("tdRange", None)

    kwargs['cmap'] = kwargs.get('cmap', 'Greens')
    colorbar = kwargs.get('colorbar', True)

    style = kwargs.pop('style', 'contour')

    figsize = kwargs.pop("figsize", (6, 4))
    layout = kwargs.pop("layout", 'constrained')

    if ax is None:
        # then generate a graph
        fig, ax = plt.subplots(figsize=figsize, layout=layout)

    xlabel = kwargs.pop("xlabel", 'Arrival Time (ms)')
    ylabel = kwargs.pop("ylabel", 'm/z')

    if style == 'contour':
        plot = ax.contourf(X, Y, Z, **kwargs)

    else:  # if style == 'grid':
        # we must provide grids to fill with the Z values
        # then the grid dimension is nr+1 x nc+1

        # for im, the bin size is constant
        dx = (X[1] - X[0]) / 2
        xx = np.zeros(len(X) + 1)
        xx[:-1] = X - dx
        xx[-1] = X[-1] + dx

        # for ms, the bin size is not constant
        yy = np.zeros(len(Y) + 1)
        yy[:-1] = Y  # copy the first values
        lastBin = Y[-1] - Y[-2]
        # last value added with the same bin as the last one
        yy[-1] = Y[-1] + lastBin

        nyy = np.zeros(len(Y) + 1)  # make a shifted copy nyy[i] = yy [i-1]
        nyy[:-1] = yy[1:]
        nyy[-1] = yy[-1] + lastBin

        # shift by half the bin size
        dy = nyy - yy
        yy = yy - dy / 2.0

        plot = ax.pcolormesh(xx, yy, Z, shading='flat', **kwargs)

    if colorbar:
        plt.colorbar(plot)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


# if __name__ == "__main__":
#
#     from dataset import fitParaAsDict
#
#     D = dataset("g13L+2_T67_s1_s25.8_d100.0_500V_20221010.hdf5")
#
#     # test waterfall
#     # levels = [2, 3, 5, 10, 15, 30, 50, 75]
#     # filelist = ["g13L+2_T67_s1_s25.8_d{:.1f}_500V_20221010.hdf5".format(i) for i in levels]
#     # datalist = []
#     # for fnam in filelist:
#     #     DD = dataset(fnam)
#     #     datalist.append(DD)
#     #     # ax = mobilogram(dataset=DD, mzRange=[1390, 1395], tdRange=[25, 35], color='red', label='exp.')
#     # # ax = plotAsWaterfall(datalist, levels=levels, tdRange = [25,35], mzRange=[1390,1395], ylabel='Trapping time (ms)', levelParam="V act.", normalize='sum')
#     # ax = plotAsWaterfall(datalist, levels=levels, tdRange = [25,35], mzRange=[1390,1395], ylabel='Trapping time (ms)', levelParam="Trap Delay", normalize='sum')
#     # ax = waterfall([D, D, D, D], levels=[1, 2, 5, 8], tdRange = [25,35], mzRange=[1390,1395], ylabel='Trapping Time (ms)')
#
#     # # test 2d map
#     map2d(D, mzRange=[1390, 1395], tdRange=[25, 35], cmap='Greys')
#     # map2d(D, mzRange=[1390, 1395], tdRange=[25, 35], cmap='Greys', style='grid')
#
#     # test mobilogram
#     ax = mobilogram(dataset=D, mzRange=[1390, 1395], tdRange=[25, 35], color='red', label='exp.')
#     D.setupConditions(m=2800.0, z=2)
#     par, err = D.fitIMPeaks([[27,28.0],[29.5,31.6]], mzRange=[1390, 1395], tdRange=[22.5, 35])
#     print(par)
#     print("parameters:\n",fitParaAsDict(par))
#     x = np.linspace(25,35, 500)
#     env, peaks = D.fitCurves(x, par)
#     ax.plot(x,env,'g-', label='fit')
#     for pk in peaks:
#         ax.plot(x, pk, 'b--')
#     ax.legend()
#     plt.show()
#
#     # x, y = D.extractIM(mzRange=[1390, 1395], tdRange=[20, 40],  tdRange=[22.5, 35])
#     # yfit = D.multiFick(x, *par)
#     # residues = y - yfit
#     # plt.plot(residues)
#
#     # x, y = D.extractIM(mzRange=[1390, 1395], tdRange=[20, 40])
#     # mobilogram(x=x, y=y, color='red', label='plain')
#
#     # test mass spec
#     ax1 = massSpec(dataset=D, mzRange=[1360, 1400], tdRange=[29.5, 32], color='blue', label='plain')
#     x, y = D.extractMS(mzRange=[1385, 1395])
#     massSpec(ax1, x=x, y=y, color='red')
#
#     plt.show()
