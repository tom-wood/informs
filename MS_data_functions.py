import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import pandas as pd

mpl.rcParams['mathtext.default'] = 'regular'
"""
mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['legend.fontsize'] = 14.0
"""
mpl.rc('ytick', #labelsize=16, 
       direction='out')
mpl.rc('xtick', #labelsize=16, 
       direction='out')
mpl.rc('xtick.major', size=8)#, width=2)
mpl.rc('xtick.minor', size=8)
mpl.rc('ytick.major', size=8)#, width=2)

def mpl_style(style):
    if style == 'presentation':
	mpl.rcParams['axes.linewidth'] = 2.0
	mpl.rcParams['lines.linewidth'] = 2.0
	mpl.rcParams['font.size'] = 16.0
	mpl.rcParams['legend.fontsize'] = 14.0
	mpl.rc('ytick', labelsize=16, direction='out')
	mpl.rc('xtick', labelsize=16, direction='out')
	mpl.rc('xtick.major', size=8, width=2)
	mpl.rc('xtick.minor', size=8)
	mpl.rc('ytick.major', size=8, width=2)

#Ionization factors (from Ar + gas mixture)
I_ar = 1.000
I_nh3 = 0.811 #Old value = 0.849
I_nd3 = 0.775 #Old value = 0.870
I_ndh2 = (2 * I_nh3 + I_nd3) / 3 #Old value = 0.856
I_nd2h = (I_nh3 + 2 * I_nd3) / 3 #Old value = 0.863
I_h2 = 0.686 #Old value = 0.568
I_d2 = 0.252
I_hd = (I_h2 + I_d2) / 2 #Old value = 0.410
I_n2 = 0.585 #Old value = 0.628

#coefficients at 100% single gas purity first
###lots more to put in, but work out nd3 sfs first.
nh3_amus = [1, 2, 14, 15, 16, 17, 18, 28]
nh3_coeffs = np.array([0.0212, 0.0162, 0.0076, 0.0213, 0.3933, 0.5099,
                       0.0159, 0.0146])
nd3_amus = [1, 2, 4, 14, 16, 17, 18, 19, 20, 21, 22, 28]
nd3_coeffs = np.array([0.0101, 0.0226, 0.0114, 0.0068, 0.0158, 0.0053,
                       0.3890, 0.0148, 0.4944, 0.0024, 0.0126, 0.0147])

def nd3_fit(nh3_coeffs, nd3_coeffs, guesses=[0.01, 1]):
    nh3_tot = 0
    for n in nh3_coeffs[2:6]:
        nh3_tot += n
    f4, f3, f2, f1 = [n / nh3_tot for n in nh3_coeffs[2:6]]
    p1 = (1 - f1) / 3
    p2 = (1 - f2 / (1 - f1)) / 2
    p3 = 1 - f3 / ((1 - f1) * (1 - f2 / (1 - f1)))
    probs = [p1, p2, p3]
    nd3_tot = 0
    for n in nd3_coeffs[3:9]:
        nd3_tot += n
    nd3_new = [n / nd3_tot for n in nd3_coeffs[3:9]]
    def residuals(guesses, probs, nd3_new):
        x, y = guesses
        p1, p2, p3 = probs
        d14, d16, d17, d18, d19, d20 = nd3_new
        err = np.zeros(len(nd3_new))
        err[0] = np.abs(d14 - (1 - x) * 6 * y**3 * p1 * p2 * p3)
        err[1] = np.abs(d16 - (1 - x) * 6 * y**2 * p1 * p2 * (1 - y * p3))
        err[2] = np.abs(d17 - 2 * x * y * p1 * (2 - y * p2 - p2))
        err[3] = np.abs(d18 - p1 * (1 - 2 * y * p2) * (3 * y * (1 - x) + x))
        err[4] = np.abs(d19 - x * (1 - p1 - 2 * y * p1))
        err[5] = np.abs(d20 - (1 - x) * (1 - 3 * y * p1))
        return err
    params = leastsq(residuals, guesses, args=(probs, nd3_new))
    p = list(params[0])
    fit14 = (1 - p[0]) * 6 * p[1]**3 * p1 * p2 * p3
    fit16 = (1 - p[0]) * 6 * p[1]**2 * p1 * p2 * (1 - p[1] * p3)
    fit17 = 2 * p[0] * p[1] * p1 * (1 - p[1] * p2 - p2)
    fit18 = p1 * (1 - 2 * p[1] * p2) * (3 * p[1] * (1 - p[0]) + p[0])
    fit19 = p[0] * (1 - p1 - 2 * p[1] * p1)
    fit20 = (1 - p[0]) * (1 - 3 * p[1] * p1)
    fits = [fit14, fit16, fit17, fit18, fit19, fit20]
    return p, fits

#nd3_fit gives the atom % of H in ND3 as 1.367 (i.e. 0.01367).
#Therefore we take out 17, 19 and 21 peaks and modify 18 peak
nd3_amus = [amu for amu in nd3_amus if amu not in [17, 19, 21]]
nd3_sfs = []
for i, c in enumerate(nd3_coeffs):
    if i not in [5, 7, 9]:
        if i == 6:
            coeff = c * (1 - (0.0399 * 2) / 3)
        else:
            coeff = c
        nd3_sfs.append(coeff)

def normalize_coeffs(coeffs):
    total = 0
    for c in coeffs:
        total += c
    new_coeffs = []
    for c in coeffs:
        new_coeffs.append(float(c) / total)
    return new_coeffs

nd3_sfs = normalize_coeffs(nd3_sfs)
nd3_sfs = [(nd3_amus[i], s) for i, s in enumerate(nd3_sfs)]
nh3_sfs = [(nh3_amus[i], s) for i, s in enumerate(nh3_coeffs)]

#now work out ndh2 and nd2h sfs
y = nd3_fit(nh3_coeffs, nd3_coeffs)[0][1]
nh3_tot = 0
for n in nh3_coeffs[2:6]:
    nh3_tot += n
f4, f3, f2, f1 = [n / nh3_tot for n in nh3_coeffs[2:6]]
p1 = (1 - f1) / 3
p2 = (1 - f2 / (1 - f1)) / 2
p3 = 1 - f3 / ((1 - f1) * (1 - f2 / (1 - f1)))

ndh2_temp_sfs = [6 * y * p1 * p2 * p3, 4 * y * p1 * p2 * (1 - p3),
                 y * p1 * (1 - 2 * p2) + 2 * p1 * p2 * (1 - y * p3),
                 2 * p1 * (1 - y * p2 - p2), 1 - 2 * p1 - y * p1, 
                 0, 0, 0, 0, 0]
nd2h_temp_sfs = [6 * y**2 * p1 * p2 * p3, 2 * y**2 * p1 * p2 * (1 - p3),
                 4 * y * p1 * p2 * (1 - y * p3), 
                 2 * y * p1 * (1 - p2 - y * p2), p1 * (1 - 2 * y * p2),
                 1 - p1 - 2 * y * p1, 0, 0, 0, 0]
ex_H = nh3_sfs[-2][1] / nh3_sfs[-3][-1]
ex_D = nd3_sfs[-2][1] / nd3_sfs[-3][-1]

#other gases:
h2_sfs = [(1, 0.3580), (2, 0.6349), (3, 0.0071)]
d2_sfs = [(1, 0.0377), (2, 0.0070), (3, 0.0039), (4, 0.9403), (6, 0.0111)]
n2_sfs = [(14, 0.0530), (28, 0.9401), (29, 0.0069)]
ar_sfs = [(20, 0.1133), (36, 0.0030), (40, 0.8836)]

def styles():
    """Return default colours for plots"""
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'sienna', 'slateblue',
               'dimgrey', 'maroon', 'teal', 'hotpink', 'indigo', 'crimson',
               'chartreuse']
    return colours

def mspec_plot(time, traces, labels, legend=True, legend_loc=0,
               xlabel='Time / min', ylabel='Pressure / mbar',
               xlim_left=None, xlim_right=None, ylim_bottom=None,
               ylim_top=None, fig_size=(10, 8), colours=None,
               tight_layout=True, dpi=None):
    """
    Produce filled cumulative plot of deuterated (or otherwise) species

    Args:
        time: array or list of arrays with time(s) values
        traces: array or list of arrays with traces
        legend (bool): determines presence of legend
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis
        xlim_left: x-axis lefthand limit---None by default
        xlim_right: x-axis righthand limit---None by default
        ylim_bottom: y-axis lower limit---None by default
        ylim_top: y-axis upper limit---None by default
        fig_size (tup): determines dimensions of figure
        colours (list): list of colours if defaults not wanted
    """
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tick_params(top=False, right=False)
    if type(colours) == type(None):
        colours=styles()
    if type(labels) == type(''):
        labels = [labels]
    if type(traces) == type(np.array([])) and \
       type(time) == type(np.array([])):
        if time.ndim != 1:
            for index, t in enumerate(time[0, :]):
                if traces.ndim == 1 or traces.shape[-1] == 1:
                    ax.plot(time[:, index], traces, label=labels[index],
                            color=colours[index])
                else:
                    ax.plot(time[:, index], traces[:, index], 
                            label=labels[index], color=colours[index])
        else:
            if traces.ndim == 1 or traces.shape[-1] == 1:
                ax.plot(time, traces, label=labels[0], color=colours[0])
            else:
                for index, trace in enumerate(traces[0, :]):
                    ax.plot(time, traces[:, index], label=labels[index],
                            color=colours[index])
    elif type(traces) == type(np.array([])) and \
         type(time) != type(np.array([])):
        for index, t in enumerate(time):
            if traces.ndim == 1 or traces.shape[-1] == 1:
                ax.plot(t, traces, label=labels[index], 
                        color=colours[index])
            else:
                ax.plot(t, traces[:, index], label=labels[index], 
                        color=colours[index])
    elif type(traces) != type(np.array([])) and \
         type(time) == type(np.array([])):
        for index, trace in enumerate(traces):
            if time.ndim == 1 or time.shape[-1] == 1:
                ax.plot(time, trace, label=labels[index], 
                        color=colours[index])
            else:
                ax.plot(time[:, index], trace, label=labels[index], 
                        color=colours[index])
    else:
        for index, trace in enumerate(traces):
            ax.plot(time[index], trace, label=labels[index],
                    color=colours[index])
    if legend:
        plt.legend(loc=legend_loc)
    if type(xlim_left) == int or type(xlim_left) == float:
        ax.set_xlim(left=float(xlim_left))
    if type(xlim_right) == int or type(xlim_right) == float:
        ax.set_xlim(right=float(xlim_right))
    if type(ylim_top) == int or type(ylim_top) == float:
        ax.set_ylim(top=float(ylim_top))
    if type(ylim_bottom) == int or type(ylim_bottom) == float:
        ax.set_ylim(bottom=float(ylim_bottom))
    ax.yaxis.get_major_formatter().set_powerlimits((-3, 4))
    if tight_layout:
        plt.tight_layout()

def mspec_fit(time, traces, fits, labels, legend=True, 
              legend_loc='upper right', xlabel='Time / min', 
              ylabel='Pressure / %', xlim_left=None, xlim_right=None, 
              ylim_bottom=None, ylim_top=None, fig_size=(10, 8),
              fit_legend=False, msize=10, open_circles=False, 
              colours=None, linewidth=None, linestyle='dotted',
              opacity=1, dpi=None):
    """
    Produce filled cumulative plot of deuterated (or otherwise) species

    Args:
        time: array or list of arrays with time(s) values
        traces: array or list of arrays with traces
        legend (bool): determines presence of legend
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis
        xlim_left: x-axis lefthand limit---None by default
        xlim_right: x-axis righthand limit---None by default
        ylim_bottom: y-axis lower limit---None by default
        ylim_top: y-axis upper limit---None by default
        fig_size (tup): determines dimensions of figure
        fit_legend: determines whether fit lines are included in legend
        msize: determines the size of markers
        open_circles (bool): determines presence of open circles or not
        colours (list): list of colours if defaults not wanted
        linewidth: value in pts for line width
        linestyle: 'dotted', 'dashed', 'solid'
        opacity: value between 0 and 1 for scatter points' opacity or
        list of opacities.
    """
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tick_params(top=False, right=False)
    if type(colours) == type(None):
        colours=styles()
    #make sure everything is in lists
    if type(time) == type(np.array([])):
        if time.ndim == 1:
            times = [time]
        else:
            times = [time[:, index] for index, t in enumerate(time[0, :])]
    else:
        times = time[:]
    if type(traces) == type(np.array([])):
        if traces.ndim == 1:
            traces = [traces]
        else:
            new_traces = [traces[:, index] for index, t in 
                          enumerate(traces[0, :])]
            traces = new_traces
    if type(fits) == type(np.array([])):
        if fits.ndim == 1:
            fits = [fits]
        else:
            new_fits = [fits[:, index] for index, f in 
                          enumerate(fits[0, :])]
            fits = new_fits
    #now check that times is double length of traces
    if len(times) != len(traces) + len(fits):
        if len(times) == len(traces):
            times += times
        elif len(times) < len(traces):
            for n in range(len(traces[1:]) + len(fits)):
                times.append(times[0])
    #make labels a list
    if type(labels) == type(''):
        labels = [labels]
    if type(opacity) == type(1) or type(opacity) == type(0.1):
        opacity = [opacity for trace in traces]
    #now plot
    half_t = len(times) / 2
    for index, t in enumerate(times[:half_t]):
        if open_circles:
            ax.scatter(t, traces[index], s=msize, facecolors='none',
                       label=labels[index], edgecolors=colours[index],
                       alpha=opacity[index])
        else:
            ax.scatter(t, traces[index], color=colours[index], s=msize,
                       label=labels[index], alpha=opacity[index])
        fit_label = labels[index] + ' fit'
        if linewidth:
            ax.plot(times[index + len(traces)], fits[index], lw=linewidth,
                    color=colours[index], ls=linestyle, label=fit_label)
        else:
            ax.plot(times[index + len(traces)], fits[index], 
                    color=colours[index], ls=linestyle, label=fit_label)
    if legend:
        ax.legend(loc=legend_loc)
    #legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if fit_legend:
            handles_half = len(handles) / 2
            handles = handles[:handles_half]
            ax.legend(handles, labels, loc=legend_loc)
        else:
            ax.legend(handles, labels, loc=legend_loc)
    if type(xlim_left) == int or type(xlim_left) == float:
        ax.set_xlim(left=float(xlim_left))
    if type(xlim_right) == int or type(xlim_right) == float:
        ax.set_xlim(right=float(xlim_right))
    if type(ylim_top) == int or type(ylim_top) == float:
        ax.set_ylim(top=float(ylim_top))
    if type(ylim_bottom) == int or type(ylim_bottom) == float:
        ax.set_ylim(bottom=float(ylim_bottom))
    ax.yaxis.get_major_formatter().set_powerlimits((-3, 4))
    plt.tight_layout()

def contourf(times, amus, ps, color_num=20, fig_size=(10, 8), 
             xlabel='Time / min', ylabel='m/z / a.m.u.',
             zlabel='Gas Fraction', dpi=None):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_subplot(111)
    plt.tick_params(top=False, right=False)
    plt.contourf(times, amus, ps, color_num)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar()
    cbar.set_label(zlabel, rotation=270, labelpad=20)
    plt.tight_layout()

def temp_fs(sfs, mzs):
    """Return list of normalized sensitivity factors for given mzs

    Args:
        sfs: list of tuples of sensitivity factors
        mzs: list of mz values
    """
    result = []
    sfs_ms = [s[0] for s in sfs]
    for m in mzs:
        if m in sfs_ms:
            i = sfs_ms.index(m)
            result.append(sfs[i][1])
        else:
            result.append(0)
    result = normalize_coeffs(result)
    return result

#for just H, the mzs used should be 2, 3, 14, 15, 16, 17, 20, 28, 36, 40
def fit_all_justhar(M, sfs, Is, fit_array=True, guesses=None):
    """Return the best fit (LM) for MS data assuming all hydrogenated

    Args:
        M: array of measured MS fractions (normalized)
        sfs: list of sensitivity factors for fragmentations of molecules in
        molecular weight order.
        Is: ionization factors in molecular weight order
        fit_array (bool): determines whether to return fitted array or not
        guesses: list of guesses for fraction of Ar, N2, NH3, H2
    Returns:
    """
    h2_s, nh3_s, n2_s, ar_s = sfs
    I_h2, I_nh3, I_n2, I_ar = Is
    #work out initial guesses
    if type(guesses) == type(None):
        g_ar = M[9] / ar_s[9]
        if g_ar > 1:
            g_ar = 1
        elif g_ar < 0:
            g_ar = 0
        g_n2 = M[7] / n2_s[7]
        if g_n2 > 1:
            g_n2 = 1
        elif g_n2 < 0:
            g_n2 = 0
        g_nh3 = M[5] / nh3_s[5]
        if g_nh3 > 1:
            g_nh3 = 1
        elif g_nh3 < 0:
            g_nh3 = 0
        g_h2 = 1 - g_n2 - g_nh3 - g_ar
        guesses = [g_ar, g_n2, g_nh3, g_h2]
    else:
        g_ar, g_n2, g_nh3, g_h2 = guesses
    if fit_array:
        nh3_s1 = np.array(nh3_s[:])
        n2_s1 = np.array(n2_s[:])
        ar_s1 = np.array(ar_s[:])
        h2_s1 = np.array(h2_s[:])
        guess_M = g_n2 * n2_s1 + g_nh3 * nh3_s1 + g_h2 * h2_s1 + g_ar \
                  * ar_s1
    def residuals(guesses, sfs, Is, M):
        h2_s, nh3_s, n2_s, ar_s = sfs
        I_h2, I_nh3, I_n2, I_ar = Is
        g_ar, g_n2, g_nh3, g_h2 = guesses
        g_ar, g_n2, g_nh3, g_h2 = normalize_coeffs([g_ar, g_n2, g_nh3,
                                                    g_h2])
        nh3_s1 = np.array(nh3_s[:])
        n2_s1 = np.array(n2_s[:])
        ar_s1 = np.array(ar_s[:])
        h2_s1 = np.array(h2_s[:])
        M_calc = g_n2 * n2_s1 + g_nh3 * nh3_s1 + g_h2 * h2_s1 + g_ar *\
                 ar_s1
        err = np.abs(M - M_calc)
        return err
    params = leastsq(residuals, guesses, args=(sfs, Is, M))
    p = list(params[0])
    #Now adjust for ionization factors
    p = [p[0] / I_ar, p[1] / I_n2, p[2] / I_nh3, p[3] / I_h2]
    p = normalize_coeffs(p)
    if fit_array:
        g_ar, g_n2, g_nh3, g_h2 = p
        nh3_s1 = np.array(nh3_s[:])
        n2_s1 = np.array(n2_s[:])
        ar_s1 = np.array(ar_s[:])
        h2_s1 = np.array(h2_s[:])
        fit_M = g_n2 * n2_s1 + g_nh3 * nh3_s1 + g_h2 * h2_s1 + g_ar * ar_s1
        return guesses, p, guess_M, fit_M
    return guesses, p

def extract_mshist_pd(csv, bins=None, sums=None):
    """Return times, amus and pressures for csv file array"""
    cyc_len = int(csv['mass amu'].max())
    cyc_num = int(csv.shape[0] / cyc_len)
    if bins:
        times = np.zeros((cyc_len, cyc_num / bins))
        amus = np.zeros((cyc_len, cyc_num / bins))
        ps = np.zeros((cyc_len, cyc_num / bins))
        for n in range(cyc_num / bins):
            times[:, n] = csv['ms'][int(cyc_len * n * bins):
                                    int(cyc_len * (n * bins + 1))] / 6e4
            amus[:, n] = csv['mass amu'][int(cyc_len * n * bins):
                                         int(cyc_len * (n * bins + 1))]
            ps[:, n] = csv['Faraday torr'][int(cyc_len * n * bins):
                                           int(cyc_len * (n * bins + 1))]
        return times, amus, ps
    if sums:
        times = np.zeros((cyc_len, cyc_num / sums))
        amus = np.zeros((cyc_len, cyc_num / sums))
        ps = np.zeros((cyc_len, cyc_num / sums))
        for n in range(cyc_num / sums):
            times[:, n] = np.mean(csv['ms'].values[\
                int(cyc_len * n * sums):\
                int(cyc_len * (n + 1) * sums)].reshape(sums, cyc_len),\
                                 axis=0) / 6e4
            amus[:, n] = csv['mass amu'][int(cyc_len * n * sums):
                                         int(cyc_len * (n * sums + 1))]
            ps[:, n] = np.mean(csv['Faraday torr'].values[\
                int(cyc_len * n * sums):\
                int(cyc_len * (n + 1) * sums)].reshape(sums, cyc_len),\
                              axis=0)
        return times, amus, ps
    else:
        times = np.zeros((cyc_len, cyc_num))
        amus = np.zeros((cyc_len, cyc_num))
        ps = np.zeros((cyc_len, cyc_num))
        for n in range(cyc_num):
            times[:, n] = csv.values[cyc_len * n:cyc_len * (n + 1), 1] / 6e4
            amus[:, n] = csv.values[cyc_len * n:cyc_len * (n + 1), 2]
            ps[:, n] = csv.values[cyc_len * n:cyc_len * (n + 1), 3]
    return times, amus, ps

def get_gasfracs(ps):
    tot_ps = np.sum(ps, axis=0)
    tot_ps = np.meshgrid(tot_ps, np.zeros(ps.shape[0]))[0]
    fracs = ps / tot_ps
    return fracs
 
def fit_MS_data(times, fracs, time_range=None):
    mzs2 = [2, 3, 14, 15, 16, 17, 28, 36, 40]
    nh3_sfs2 = temp_fs(nh3_sfs, mzs2)
    I2_nh3 = I_nh3 * (1 - nh3_sfs[0][1] - nh3_sfs[6][1])
    n2_sfs2 = temp_fs(n2_sfs, mzs2)
    I2_n2 = I_n2 * (1 - n2_sfs[2][1])
    ar_sfs2 = temp_fs(ar_sfs, mzs2)
    I2_ar = I_ar
    h2_sfs2 = temp_fs(h2_sfs, mzs2)
    I2_h2 = I_h2 * (1 - h2_sfs[0][1])
    I2_h2 /= 1.3
    sfs2 = [h2_sfs2, nh3_sfs2, n2_sfs2, ar_sfs2]
    Is2 = [I2_h2, I2_nh3, I2_n2, I2_ar]
    if type(time_range) == type([]) or type(time_range) == type((0,)):
	t1, t2 = [np.searchsorted(times[0, :], tval) for tval in time_range]
	times2 = times[0, t1:t2]
    else:
	times2 = times[0, :]
	t1, t2 = 0, times.shape[1]
    M2 = np.row_stack((fracs[1:3, t1:t2], fracs[13:17, t1:t2], 
		       fracs[27, t1:t2], fracs[35, t1:t2], fracs[39, t1:t2])) 
    norm_M2 = np.zeros(M2.shape)
    M2_tot = np.sum(M2, axis=0)
    for i, col in enumerate(M2[0, :]):
	norm_M2[:, i] = M2[:, i] / M2_tot[i]
    fit_M2 = np.zeros(norm_M2.shape).T
    params2 = np.zeros((norm_M2.shape[1], 4))
    for i, col in enumerate(norm_M2[0, :]):
	if i == 0:
	    g, params2[i, :], gM, fit_M2[i, :] = \
		    fit_all_justhar(norm_M2[:, i], sfs2, Is2,
				    guesses=[1, 0, 0, 0])
	else:
	    g, params2[i, :], gM, fit_M2[i, :] = \
		    fit_all_justhar(norm_M2[:, i], sfs2, Is2, guesses=g)
    params2[params2 < -0.05] = np.nan
    params2[params2 > 1.05] = np.nan
    params2[np.isnan(params2[:, 0]), :] = np.array([np.nan, np.nan, np.nan,
						    np.nan])
    for i, b in enumerate(params2):
	if np.isnan(b).any():
	    if i == 0:
		params2[i] = np.array([1, 0, 0, 0])
	    else:
		params2[i] = params2[i - 1]
    return times2, params2

def get_log_data(log_fpath):
    log_data = pd.read_csv(log_fpath, sep='\t', 
	                   usecols=[4, 8, 9, 10, 11, 12, 15, 16, 17],
			   names=['Time', 'MFM', 'MFC1_set', 'MFC1',
			          'MFC2_set', 'MFC2', 'Main', 'Main_set',
				  'Aux'], skiprows=1)
    return log_data
