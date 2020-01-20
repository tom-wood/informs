########Deuterated fitting functions not working yet (not started)
#version 0.3.0 alpha: introduced Experiment class, partial so far.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import pandas as pd
import scipy.optimize as opt

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
plt.rcParams['image.cmap'] = 'afmhot'

def mpl_style(style, size=20.0):
    if style == 'presentation':
        mpl.rcParams['axes.linewidth'] = 2.0
        mpl.rcParams['lines.linewidth'] = 2.0
        mpl.rcParams['font.size'] = size
        mpl.rcParams['axes.labelsize'] = size
        mpl.rcParams['legend.fontsize'] = size
        mpl.rc('ytick', labelsize=size, direction='out')
        mpl.rc('xtick', labelsize=size, direction='out')
        mpl.rc('xtick.major', size=8, width=2)
        mpl.rc('xtick.minor', size=8)
        mpl.rc('ytick.major', size=8, width=2)

#Ionization factors (from Ar + gas mixture)
class IonizationFactors:
    def __init__(self, I_ar=None, I_nh3=None, I_nd3=None, I_h2=None, I_d2=None,
                 I_n2=None, fname=None, sep='\t'):
        self.all_Is = {'ar': I_ar, 'nh3': I_nh3, 'nd3': I_nd3, 'h2': I_h2,
                       'd2': I_d2, 'n2': I_n2}
        self.labels = {'ar': 'Ar', 'nh3': 'NH3', 'nd3': 'ND3', 'h2': 'H2',
                       'd2': 'D2', 'n2': 'N2'}
        self.original_Is = self.all_Is.copy()
        self.reset()
        if fname:
            self.load_factors(fname, sep=sep)
        
    def __str__(self):
        s = "Ionization factors:"
        s += ''.join([f"\n{self.labels[k]}\t{v:.3f}" for k, v in 
                      self.all_Is.items() if v is not None])
        return s
     
    def reset(self):
        defaults = {'ar': 1., 'nh3': 0.811, 'nd3': 0.775, 'h2': 0.686, 
                    'd2':0.252, 'n2': 0.585}
        for k, v in self.all_Is.items():
            if v is None:
                if k not in defaults.keys():
                    continue
                self.all_Is.update({k: defaults[k]})
            if k not in self.original_Is.keys():
                continue
            elif v != self.original_Is[k]:
                self.all_Is.update({k: self.original_Is[k]})
        self.all_Is.update({'ndh2': (2 * self.all_Is['nh3'] + \
                                     self.all_Is['nd3']) / 3,
                            'nd2h': (self.all_Is['nh3'] + 2 * \
                                     self.all_Is['nd3']) / 3,
                            'hd': (self.all_Is['h2'] + self.all_Is['d2']) / 2})
        self.labels.update({'ndh2': 'NDH2', 'nd2h': 'ND2H', 'hd': 'HD'})
    
    def save_factors(self, fname):
        with open(fname, 'w') as f:
            for k, v in self.all_Is.items():
                f.write(f'{k}\t{v:.4f}\n')
    
    def load_factors(self, fname, sep='\t'):
        with open(fname, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                k, v = line.strip().split(sep)
                self.all_Is.update({k: float(v)})
        self.original_Is = self.all_Is.copy()

#coefficients at 100% single gas purity first
class FragmentationRatios:
    def __init__(self, ar_rats=None, nh3_rats=None, nd3_rats=None, h2_rats=None,
                 d2_rats=None, n2_rats=None, fname=None, sep='\t'):
        self.all_rats = {'ar': ar_rats, 'nh3': nh3_rats, 'nd3': nd3_rats,
                         'h2': h2_rats, 'd2': d2_rats, 'n2': n2_rats}
        self.labels = {'ar': 'Ar', 'nh3': 'NH3', 'nd3': 'ND3', 'h2': 'H2',
                       'd2': 'D2', 'n2': 'N2'}
        self.reset()
        if fname:
            self.load_ratios(fname, sep)
    
    def __str__(self):
        s = ''
        for k, v in self.all_rats.items():
            if v is None:
                continue
            else:
                s += f"\n\n{self.labels[k]}"
                s += ''.join([f'\n{m}\t{r}' for m, r in zip(*v)])
        if s:
            s = s[1:]
        return s
    
    def reset(self):
        defaults = {'ar': ([20, 36, 40], [0.1133, 0.0030, 0.8836]),
                    'h2': ([1, 2, 3], [0.3580, 0.6349,0.0071]),
                    'd2': ([1, 2, 3, 4, 6], 
                           [0.0377, 0.0070, 0.0039, 0.9403, 0.0111]),
                    'n2': ([14, 28, 29], [0.0530, 0.9401, 0.0069]),
                    'nh3': ([1, 2, 14, 15, 16, 17, 18, 28],
                            np.array([0.0212, 0.0162, 0.0076, 0.0213, 0.3933,
                                      0.5099, 0.0159, 0.0146])),
                    'nd3': ([1, 2, 4, 14, 16, 17, 18, 19, 20, 21, 22, 28], 
                            np.array([0.0101, 0.0226, 0.0114, 0.0068, 0.0158,
                                      0.0053, 0.3890, 0.0148, 0.4944, 0.0024, 
                                      0.0126, 0.0147]))}
        for k, v in self.all_rats.items():
            if v is None:
                self.all_rats.update({k: defaults[k]})
        self.all_sfs = {k:(list(zip(*v)) if v is not None else v) for k, v in self.all_rats.items()}
        if self.all_rats['nh3'] is not None and self.all_rats['nd3'] is not None:
            nd3_amus = [amu for amu in self.all_rats['nd3'][0] if amu not in 
                        [17, 19, 21]]
            self.all_sfs['nd3'] = []
            for i, c in enumerate(self.all_rats['nd3'][1]):
                if i not in [5, 7, 9]:
                    if i == 6:
                        coeff = c * (1 - (0.0399 * 2) / 3)
                    else:
                        coeff = c
                    self.all_sfs['nd3'].append(coeff)
            self.all_sfs['nd3'] = self.normalize_coeffs(self.all_sfs['nd3'])
            self.all_sfs['nd3'] = [(nd3_amus[i], s) for i, s in 
                        enumerate(self.all_sfs['nd3'])]
            self.all_sfs['nh3'] = [(self.all_rats['nh3'][0][i], s) for i, s in 
                             enumerate(self.all_rats['nh3'][1])]
            y = self.nd3_fit(self.all_rats['nh3'][1], self.all_rats['nd3'][1])[0][1]
            nh3_tot = 0
            for n in self.all_rats['nh3'][1][2:6]:
                nh3_tot += n
            f4, f3, f2, f1 = [n / nh3_tot for n in self.all_rats['nh3'][1][2:6]]
            p1 = (1 - f1) / 3
            p2 = (1 - f2 / (1 - f1)) / 2
            p3 = 1 - f3 / ((1 - f1) * (1 - f2 / (1 - f1)))
            self.all_sfs['ndh2'] = [6 * y * p1 * p2 * p3, 4 * y * p1 * p2 * (1 - p3),
                                  y * p1 * (1 - 2 * p2) + 2 * p1 * p2 * (1 - y * p3),
                                  2 * p1 * (1 - y * p2 - p2), 1 - 2 * p1 - y * p1, 
                                  0, 0, 0, 0, 0]
            self.all_sfs['nd2h'] = [6 * y**2 * p1 * p2 * p3, 2 * y**2 * p1 * p2 * (1 - p3),
                                  4 * y * p1 * p2 * (1 - y * p3), 
                                  2 * y * p1 * (1 - p2 - y * p2), p1 * (1 - 2 * y * p2),
                                  1 - p1 - 2 * y * p1, 0, 0, 0, 0]
            self.ex_H = self.all_sfs['nh3'][-2][1] / self.all_sfs['nh3'][-3][-1]
            self.ex_D = self.all_sfs['nd3'][-2][1] / self.all_sfs['nd3'][-3][-1]
            
    def normalize_coeffs(self, coeffs):
        total = 0
        for c in coeffs:
            total += c
        new_coeffs = []
        for c in coeffs:
            new_coeffs.append(float(c) / total)
        return new_coeffs
    
    def nd3_fit(self, nh3_coeffs, nd3_coeffs, guesses=[0.01, 1]):
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
    
    def self_calibrate(self, gas_type, exp, x_range):
        """Update the fragmentation ratios for a gas_type for current data
        
        Args:
            gas_type (str): can (currently) be 'Ar', 'NH3', 'H2', 'N2'
            exp: Experiment instance with MS times, amus and fracs defined
            x_range (list): start and stop times to calibrate over; if more than
            one set of times, then order goes s0, e0, s1, e1 etc. where si is
            start time and ei is end time.
        """
        times, amus, fracs = exp.MS_times, exp.MS_amus, exp.MS_fracs
        x_starts = [np.searchsorted(times[0, :], x) for x in x_range[::2]]
        x_ends = [np.searchsorted(times[0, :], x) for x in x_range[1::2]]
        #work out mz means between those times:
        fracs_reduced = fracs[:, x_starts[0]:x_ends[0]]
        if len(x_starts) > 1:
            for i, xs in enumerate(x_starts[1:]):
                fracs_reduced = np.column_stack((fracs_reduced,
                                                fracs[:, xs:x_ends[1:][i]]))
        fracs_av = np.mean(fracs_reduced, axis=1)
        key = gas_type.lower()
        if key in self.all_sfs.keys():
            new_coeffs = []
            for sf in self.all_sfs[key]:
                av_frac = fracs_av[int(np.where(amus[:, 0] == sf[0])[0])]
                new_coeffs.append(av_frac)
            new_coeffs = self.normalize_coeffs(new_coeffs)
            new_sfs = [(sf[0], new_coeffs[i]) for i, sf in enumerate(self.all_sfs[key])]
            self.all_sfs[key] = new_sfs
        else:
            print("Can't reset sensitivity factors for gas %s." % gas_type)
    
    def save_ratios(self, fname):
        with open(fname, 'w') as f:
            for l, r in self.all_rats.items():
                if r is None:
                    continue
                f.write(f"{l}\n")
                for amu, frac in zip(*r):
                    f.write(f"{amu}\t{frac}\n")
    
    def load_ratios(self, fname, sep='\t'):
        """Load previously saved fragmentation ratios
        
        Args:
            fname: filepath to saved ratios
            sep: separator for saved ratios (defaults to tab)
        """
        with open(fname, 'r') as f:
            sfs = []
            key = None
            for line in f:
                if line.strip() in self.all_rats.keys():
                    if key:
                        self.all_sfs[key] = sfs
                        self.all_rats[key] = list(zip(*sfs))
                    key = line.strip()
                    sfs = []
                else:
                    sfs.append(line.strip().split(sep))
                    sfs[-1] = (int(sfs[-1][0]), float(sfs[-1][1]))
            self.all_sfs[key] = sfs
            self.all_rats[key] = list(zip(*sfs))
                    

all_sfs = FragmentationRatios(ar_rats=([20, 36, 40], [0.1133, 0.0030, 0.8836]),
                              h2_rats=([1, 2, 3], [0.3580, 0.6349,0.0071]),
                              d2_rats=([1, 2, 3, 4, 6], 
                                       [0.0377, 0.0070, 0.0039, 0.9403, 0.0111]),
                              n2_rats=([14, 28, 29], [0.0530, 0.9401, 0.0069]),
                              nh3_rats=([1, 2, 14, 15, 16, 17, 18, 28],
                                        np.array([0.0212, 0.0162, 0.0076, 
                                                  0.0213, 0.3933, 0.5099, 
                                                  0.0159, 0.0146])),
                              nd3_rats=([1, 2, 4, 14, 16, 17, 18, 19, 20, 21, 22, 
                                        28], np.array([0.0101, 0.0226, 0.0114, 
                                                       0.0068, 0.0158, 0.0053,
                                                       0.3890, 0.0148, 0.4944, 
                                                       0.0024, 0.0126, 0.0147])))
                                        
                            

class Experiment:
    def __init__(self, fpath="", I_factors=None, sfs=None):
        self.csv_fpath = ""
        self.log_fpath = ""
        self.save_fpath = ""
        self.offsets = {"MFC1" : 0, "MFC2" : 0, "MFM" : 0}
        self.MS_data = None
        self.MS_fit_times = None
        self.MS_fit_params = None
        self.MS_fit_labels = None
        self.log_data = None
        self.fpath = fpath
        if sfs is None:
            self.all_sfs = FragmentationRatios()
        else:
            self.all_sfs = all_sfs
        if I_factors is None:
            self.Is = IonizationFactors()
        else:
            self.Is = I_factors
        self.CRD = False
        self.Tc_indices = None
        self.av_eq_T = None
        self.conv = None
        self.conv_std = None
        self.conv2 = None
        self.conv2_unc = None
        self.Tfit = None
        self.conv_fit = None
        self.conv_single_params = None
        self.conv_single_cov = None
        self.conv_single_std = None
        self.conv_single_Cfit = None
        self.conv_double_corr = None
        self.conv_double_params = None
        self.conv_double_cov = None
        self.conv_double_std = None
        self.conv_double_Cfit = None
        self.conv_double_corr = None
    
    def assign_csv_fpath(self, fname):
        """Assign a filepath/name for the csv file"""
        self.csv_fpath = self.fpath + fname
    
    def assign_log_fpath(self, fname):
        """Assign a filepath/name for the log file"""
        self.log_fpath = self.fpath + fname
    
    def assign_offsets(self, strings, vals):
        """Assign offsets as a dictionary
        
        Args:
            strings: list of strings or single string; should take value of 
            "MFC1", "MFC2", "MFM".
            vals: list of values or single value corresponding to s
        """
        if type(strings) == str:
            strings = [strings]
        self.offsets.update(zip(strings, vals))
    
    def extract_MS_data(self, sums=None):
        """Extract MS data into times, amus and fracs
        
        Args:
            sums (int): number of histogram instances to sum (take mean) over
        """
        csv = pd.read_csv(self.csv_fpath, sep=',', header=29, 
                                   usecols=[0, 2, 3, 4])
        cyc_len = int(csv['mass amu'].max())
        cyc_num = int(csv.shape[0] / cyc_len)
        if sums:
            times = np.zeros((cyc_len, cyc_num // sums))
            amus = np.zeros((cyc_len, cyc_num // sums))
            ps = np.zeros((cyc_len, cyc_num // sums))
            t = csv['ms'].values
            p = csv['Faraday torr'].values
            for n in range(cyc_num // sums):
                times[:, n] = np.mean(t[int(cyc_len * n * sums):\
                    int(cyc_len * (n + 1) * sums)].reshape(sums, cyc_len),\
                                     axis=0) / 6e4
                amus[:, n] = csv['mass amu'][int(cyc_len * n * sums):
                                             int(cyc_len * (n * sums + 1))]
                ps[:, n] = np.mean(p[int(cyc_len * n * sums):\
                    int(cyc_len * (n + 1) * sums)].reshape(sums, cyc_len),\
                                  axis=0)
        else:
            times = np.zeros((cyc_len, cyc_num))
            amus = np.zeros((cyc_len, cyc_num))
            ps = np.zeros((cyc_len, cyc_num))
            csv_vals = csv.values
            for n in range(cyc_num):
                times[:, n] = csv_vals[cyc_len * n:cyc_len * (n + 1), 1] / 6e4
                amus[:, n] = csv_vals[cyc_len * n:cyc_len * (n + 1), 2]
                ps[:, n] = csv_vals[cyc_len * n:cyc_len * (n + 1), 3]
        tot_ps = np.sum(ps, axis=0)
        tot_ps = np.meshgrid(tot_ps, np.zeros(ps.shape[0]))[0]
        self.MS_times = times
        self.MS_amus = amus
        self.MS_ps = ps
        self.MS_fracs = ps / tot_ps
    
    def MS_contour(self, colour_num=50, figsize=(10, 8), xlabel='Time / min',
                   ylabel='m/z / a.m.u.', zlabel='Gas Fraction', dpi=None,
                   zscale='linear', t_range=None):
        """Return filled contour plot of raw MS gas fraction data over time
        
        Args:
            colour_num (int): number of colours to use on z scale
            figsize (tuple): (m, n) size of figure in inches
            xlabel (str): label for x axis
            ylabel (str): label for y axis
            zlabel (str): label for z axis
            dpi (int): image resolution
            zscale (str): can take values of 'linear', 'sqrt' or 'log', where
            the log is natural log.
            t_range (list): list of first and last time values to plot over
        Returns:
            fig: plt.figure instance
            ax: plt.axis instance
            cbar: colourbar instance
        """
        t, a, f = self.MS_times, self.MS_amus, self.MS_fracs
        if t_range is not None:
            ti0, ti1 = np.searchsorted(t[0], t_range)
            t, a, f = t[:, ti0:ti1], a[:, ti0:ti1], f[:, ti0:ti1]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        ax.tick_params(top=False, right=False)
        if zscale == 'linear':
            im = ax.contourf(t, a, f, colour_num)
        elif zscale == 'sqrt':
            im = ax.contourf(t, a, np.sqrt(np.clip(f, 1e-5, 1)), colour_num)
            zlabel = f"$\sqrt{{{zlabel}}}$"
        elif zscale == 'log':
            im = ax.contourf(t, a, np.log(np.clip(f, 1e-5, 1)), colour_num)
            zlabel = f"log({zlabel})"
        else:
            raise ValueError("zscale must take value of 'linear', 'sqrt' or 'log'")
        cbar = plt.colorbar(im)
        cbar.set_label(zlabel, rotation=270, labelpad=20)
        fig.tight_layout()
        return fig, ax, cbar
            
    def normalize_coeffs(self, coeffs):
        total = 0
        for c in coeffs:
            total += c
        new_coeffs = []
        for c in coeffs:
            new_coeffs.append(float(c) / total)
        return new_coeffs
    
    def temp_fs(self, sfs, mzs):
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
        result = self.normalize_coeffs(result)
        return result

    #for just H, the mzs used should be 2, 3, 14, 15, 16, 17, 20, 28, 36, 40
    def fit_all_justhar(self, M, sfs, Is, fit_array=True, guesses=None):
        """Return the best fit (LM) for MS data assuming all hydrogenated
    
        Args:
            M: array of measured MS fractions (normalized)
            fit_array (bool): determines whether to return fitted array or not
            guesses: list of guesses for fraction of Ar, N2, NH3, H2
        Returns:
            guesses: array of initial guessed parameters
            p: array of fitted parameters
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
            g_ar, g_n2, g_nh3, g_h2 = self.normalize_coeffs([g_ar, g_n2, g_nh3,
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
        p = self.normalize_coeffs(p)
        if fit_array:
            g_ar, g_n2, g_nh3, g_h2 = p
            nh3_s1 = np.array(nh3_s[:])
            n2_s1 = np.array(n2_s[:])
            ar_s1 = np.array(ar_s[:])
            h2_s1 = np.array(h2_s[:])
            fit_M = g_n2 * n2_s1 + g_nh3 * nh3_s1 + g_h2 * h2_s1 + g_ar * ar_s1
            return guesses, p, guess_M, fit_M
        return guesses, p

    def fit_MS_data(self, t_range=None):
        I_factors = self.Is
        all_sfs = self.all_sfs
        times, fracs = self.MS_times, self.MS_fracs
        mzs2 = [2, 3, 14, 15, 16, 17, 28, 36, 40]
        nh3_sfs2 = self.temp_fs(all_sfs.all_sfs['nh3'], mzs2)
        I2_nh3 = I_factors.all_Is['nh3'] * (1 - all_sfs.all_sfs['nh3'][0][1] - \
                                    all_sfs.all_sfs['nh3'][6][1])
        n2_sfs2 = self.temp_fs(all_sfs.all_sfs['n2'], mzs2)
        I2_n2 = I_factors.all_Is['n2'] * (1 - all_sfs.all_sfs['n2'][2][1])
        ar_sfs2 = self.temp_fs(all_sfs.all_sfs['ar'], mzs2)
        I2_ar = I_factors.all_Is['ar']
        h2_sfs2 = self.temp_fs(all_sfs.all_sfs['h2'], mzs2)
        I2_h2 = I_factors.all_Is['h2'] * (1 - all_sfs.all_sfs['h2'][0][1])
        #I2_h2 /= 1.3
        sfs2 = [h2_sfs2, nh3_sfs2, n2_sfs2, ar_sfs2]
        Is2 = [I2_h2, I2_nh3, I2_n2, I2_ar]
        if t_range is not None:
            t1, t2 = np.searchsorted(times[0, :], t_range)
            times2 = times[0, t1:t2]
        else:
            times2 = times[0, :]
            t1, t2 = 0, times.shape[1]
        M2 = np.row_stack([fracs[n - 1, t1:t2] for n in mzs2])
        norm_M2 = np.zeros(M2.shape)
        M2_tot = np.sum(M2, axis=0)
        for i, col in enumerate(M2[0, :]):
            norm_M2[:, i] = M2[:, i] / M2_tot[i]
        fit_M2 = np.zeros(norm_M2.shape).T
        params2 = np.zeros((norm_M2.shape[1], 4))
        for i, col in enumerate(norm_M2[0, :]):
            if i == 0:
                g, params2[i, :], gM, fit_M2[i, :] = \
                self.fit_all_justhar(norm_M2[:, i], sfs2, Is2, guesses=[1, 0, 0, 0])
            else:
                g, params2[i, :], gM, fit_M2[i, :] = \
                self.fit_all_justhar(norm_M2[:, i], sfs2, Is2, guesses=g)
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
        self.MS_fit_times = times2
        self.MS_gasfracs = params2
        self.MS_labels = ['Ar', '$N_{2}$', '$NH_{3}$', '$H_{2}$']
        self.MS_species = ['Ar', 'N2', 'NH3', 'H2']
    
    def plot_MS_fits(self, legend=True, legend_loc=0, xlabel='Time / min', 
                   ylabel='Gas Fraction', figsize=(10, 8), t_range=None, 
                   dpi=None):
        """Return figure of MS fitted gas fractions
    
        Args:s
            legend (bool): determines presence of legend
            legend_loc: determines location of legend
            xlabel (str): Label of the x-axis
            ylabel (str): Label of the y-axis
            figsize (tup): (m, n) determines dimensions of figure
            t_range: time in min over which to plot
            dpi: resolution of plot
        Returns:
            fig: plt.figure instance
            ax: plt.axes instance
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        ax.tick_params(top=False, right=False)
        if t_range is None:
            t0, t1 = 0, len(self.MS_fit_times)
        else:
            t0, t1 = np.searchsorted(self.MS_fit_times, t_range)
        for p, l in zip(self.MS_gasfracs.T, self.MS_labels):
            ax.plot(self.MS_fit_times[t0:t1], p[t0:t1], label=l)
        if legend:
            ax.legend(loc=legend_loc)
        ax.set_xlim(self.MS_fit_times[t0], self.MS_fit_times[t1-1])
        fig.tight_layout()
        return fig, ax
    
    def plot_MS_ratios(self, sp0, sp1, y_range=[1, 5], plot_average=True,
                       t_range=None, figsize=(10, 7), dpi=None,
                       xlabel='Time / min', ylabel='Ratio'):
        """Plot the hydrogen:nitrogen ratio taking account of isotopes
        
        Args:
            sp0: string or list of strings of species to be the numerator
            sp1: string or list of strings of species to be the denominator
            y_range: range over which to plot the y values (to ignore divide
            by zero ranging issues).
            plot_average (bool): whether to plot a horizontal line denoting 
            the average value.
            t_range: list of two values of time in minutes over which to take
            the average ratio (if required) and to plot.
            figsize (tup): (m, n) determines dimensions of figure
            dpi: resolution of the plot
        Returns:
            fig: plt.figure instance
            ax: plt.axes instance
        """
        if type(sp0) == str:
            sp0 = [sp0.upper()]
        else:
            sp0 = [sp.upper() for sp in sp0]
        if type(sp1) == str:
            sp1 = [sp1.upper()]
        else:
            sp1 = [sp.upper() for sp in sp1]
        sp0_is = [self.MS_species.index(sp) for sp in sp0 if sp in
                  self.MS_species]
        sp1_is = [self.MS_species.index(sp) for sp in sp1 if sp in
                  self.MS_species]
        num = np.sum(np.column_stack([self.MS_gasfracs[:, i] for i in 
                                      sp0_is]), axis=1)
        den = np.sum(np.column_stack([self.MS_gasfracs[:, i] for i in 
                                      sp1_is]), axis=1)
        rat = num / den
        if t_range is not None:
            t0, t1 = np.searchsorted(self.MS_fit_times, t_range)
        else:
            t0, t1 = [0, len(self.MS_fit_times)]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        ax.plot(self.MS_fit_times[t0:t1], rat[t0:t1])
        x0, x1 = [self.MS_fit_times[t] for t in [t0, t1-1]]
        if plot_average:
            av = rat[t0:t1].mean()
            ax.axhline(av, ls='dashed', color='k')
            print(f"Average ratio value between {x0:.3f} and {x1:.3f} mins = {av:.3f}")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y_range[0], y_range[1])
        fig.tight_layout()
        return fig, ax

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

    def fit_all_justdar(M, sfs, Is, fit_array=True, guesses=None):
        """Return the best fit (LM) for MS data assuming all deuterated
    
        Args:
            M: array of measured MS fractions (normalized)
            sfs: list of sensitivity factors for fragmentations of molecules in
            molecular weight order.
            Is: ionization factors in molecular weight order
            fit_array (bool): determines whether to return fitted array or not
            guesses: list of guesses for fraction of Ar, N2, ND3, D2
        Returns:
        """
        d2_s, nd3_s, n2_s, ar_s = sfs
        I_d2, I_nd3, I_n2, I_ar = Is
        #work out initial guesses
        if type(guesses) == type(None):
            g_ar = M[11] / ar_s[11]
            if g_ar > 1:
                g_ar = 1
            elif g_ar < 0:
                g_ar = 0
            g_n2 = M[9] / n2_s[9]
            if g_n2 > 1:
                g_n2 = 1
            elif g_n2 < 0:
                g_n2 = 0
            g_nd3 = M[6] / nd3_s[6]
            if g_nd3 > 1:
                g_nd3 = 1
            elif g_nd3 < 0:
                g_nd3 = 0
            g_d2 = 1 - g_n2 - g_nd3 - g_ar
            guesses = [g_ar, g_n2, g_nd3, g_d2]
        else:
            g_ar, g_n2, g_nd3, g_d2 = guesses
        if fit_array:
            nd3_s1 = np.array(nd3_s[:])
            n2_s1 = np.array(n2_s[:])
            ar_s1 = np.array(ar_s[:])
            d2_s1 = np.array(d2_s[:])
            guess_M = g_n2 * n2_s1 + g_nd3 * nd3_s1 + g_d2 * d2_s1 + g_ar \
                      * ar_s1
        def residuals(guesses, sfs, Is, M):
            d2_s, nd3_s, n2_s, ar_s = sfs
            I_d2, I_nd3, I_n2, I_ar = Is
            g_ar, g_n2, g_nd3, g_d2 = guesses
            g_ar, g_n2, g_nd3, g_d2 = normalize_coeffs([g_ar, g_n2, g_nd3,
                                                        g_d2])
            nd3_s1 = np.array(nd3_s[:])
            n2_s1 = np.array(n2_s[:])
            ar_s1 = np.array(ar_s[:])
            d2_s1 = np.array(d2_s[:])
            M_calc = g_n2 * n2_s1 + g_nd3 * nd3_s1 + g_d2 * d2_s1 + g_ar *\
                     ar_s1
            err = np.abs(M - M_calc)
            return err
        params = leastsq(residuals, guesses, args=(sfs, Is, M))
        p = list(params[0])
        #Now adjust for ionization factors
        p = [p[0] / I_ar, p[1] / I_n2, p[2] / I_nd3, p[3] / I_d2]
        p = normalize_coeffs(p)
        if fit_array:
            g_ar, g_n2, g_nd3, g_d2 = p
            nd3_s1 = np.array(nd3_s[:])
            n2_s1 = np.array(n2_s[:])
            ar_s1 = np.array(ar_s[:])
            d2_s1 = np.array(d2_s[:])
            fit_M = g_n2 * n2_s1 + g_nd3 * nd3_s1 + g_d2 * d2_s1 + g_ar * ar_s1
            return guesses, p, guess_M, fit_M
        return guesses, p
    
    def fit_deuterated_MS_data(times, fracs, time_range=None):
        mzs2 = [2, 3, 4, 6, 14, 16, 18, 20, 28, 36, 40]
        nd3_sfs2 = temp_fs(all_sfs.nd3_sfs, mzs2)
        I2_nd3 = I_factors.I_nd3 * (1 - all_sfs.nd3_sfs[0][1] - \
                                    all_sfs.nd3_sfs[7][1])
        n2_sfs2 = temp_fs(all_sfs.n2_sfs, mzs2)
        I2_n2 = I_factors.I_n2 * (1 - all_sfs.n2_sfs[2][1])
        ar_sfs2 = temp_fs(all_sfs.ar_sfs, mzs2)
        I2_ar = I_factors.I_ar
        d2_sfs2 = temp_fs(all_sfs.d2_sfs, mzs2)
        I2_d2 = I_factors.I_d2 * (1 - all_sfs.d2_sfs[0][1])
        #I2_h2 /= 1.3
        sfs2 = [d2_sfs2, nd3_sfs2, n2_sfs2, ar_sfs2]
        Is2 = [I2_d2, I2_nd3, I2_n2, I2_ar]
        if type(time_range) == type([]) or type(time_range) == type((0,)):
            t1, t2 = [np.searchsorted(times[0, :], tval) for tval in time_range]
            times2 = times[0, t1:t2]
        else:
            times2 = times[0, :]
            t1, t2 = 0, times.shape[1]
        M2 = np.row_stack([fracs[n - 1, t1:t2] for n in mzs2])
        norm_M2 = np.zeros(M2.shape)
        M2_tot = np.sum(M2, axis=0)
        for i, col in enumerate(M2[0, :]):
            norm_M2[:, i] = M2[:, i] / M2_tot[i]
        fit_M2 = np.zeros(norm_M2.shape).T
        params2 = np.zeros((norm_M2.shape[1], 4))
        for i, col in enumerate(norm_M2[0, :]):
            if i == 0:
                g, params2[i, :], gM, fit_M2[i, :] = \
                fit_all_justdar(norm_M2[:, i], sfs2, Is2, guesses=[1, 0, 0, 0])
            else:
                g, params2[i, :], gM, fit_M2[i, :] = \
                fit_all_justdar(norm_M2[:, i], sfs2, Is2, guesses=g)
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
    
    def fit_all_MS_data(times, fracs, time_range=None):
        mzs2 = [2, 3, 4, 6, 14, 15, 16, 17, 18, 19, 20, 28, 36, 40]
        nd3_sfs2 = temp_fs(all_sfs.nd3_sfs, mzs2)
        def mod_I(I, mzs2, sfs):
            factor = 1
            for sf in sfs:
                if sf[0] not in mzs2:
                    factor -= sf[1]
            return I * factor
        
        I2_nd3 = I_factors.I_nd3 * (1 - all_sfs.nd3_sfs[0][1] - \
                                    all_sfs.nd3_sfs[7][1])
        n2_sfs2 = temp_fs(all_sfs.n2_sfs, mzs2)
        I2_n2 = I_factors.I_n2 * (1 - all_sfs.n2_sfs[2][1])
        ar_sfs2 = temp_fs(all_sfs.ar_sfs, mzs2)
        I2_ar = I_factors.I_ar
        d2_sfs2 = temp_fs(all_sfs.d2_sfs, mzs2)
        I2_d2 = I_factors.I_d2 * (1 - all_sfs.d2_sfs[0][1])
        #I2_h2 /= 1.3
        sfs2 = [d2_sfs2, nd3_sfs2, n2_sfs2, ar_sfs2]
        Is2 = [I2_d2, I2_nd3, I2_n2, I2_ar]
        if type(time_range) == type([]) or type(time_range) == type((0,)):
            t1, t2 = [np.searchsorted(times[0, :], tval) for tval in time_range]
            times2 = times[0, t1:t2]
        else:
            times2 = times[0, :]
            t1, t2 = 0, times.shape[1]
        M2 = np.row_stack([fracs[n - 1, t1:t2] for n in mzs2])
        norm_M2 = np.zeros(M2.shape)
        M2_tot = np.sum(M2, axis=0)
        for i, col in enumerate(M2[0, :]):
            norm_M2[:, i] = M2[:, i] / M2_tot[i]
        fit_M2 = np.zeros(norm_M2.shape).T
        params2 = np.zeros((norm_M2.shape[1], 4))
        for i, col in enumerate(norm_M2[0, :]):
            if i == 0:
                g, params2[i, :], gM, fit_M2[i, :] = \
                fit_all_justdar(norm_M2[:, i], sfs2, Is2, guesses=[1, 0, 0, 0])
            else:
                g, params2[i, :], gM, fit_M2[i, :] = \
                fit_all_justdar(norm_M2[:, i], sfs2, Is2, guesses=g)
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
    
    def save_MS_fits(self, save_fpath):
        save_fpath = self.fpath + save_fpath
        fits = np.column_stack((self.MS_fit_times, self.MS_gasfracs))
        np.savetxt(save_fpath, fits, delimiter=',')
        return
    
    def load_MS_fits(self, save_fpath, deuterated=False):
        save_fpath = self.fpath + save_fpath
        fits = np.loadtxt(save_fpath, delimiter=',')
        self.MS_fit_times = fits.T[0]
        self.MS_gasfracs = fits[:, 1:]
        if self.MS_gasfracs.shape[1] == 4 and deuterated:
            self.MS_labels = ['Ar', '$N_{2}$', '$ND_{3}$', '$D_{2}$']
            self.MS_species = ['Ar', 'N2', 'ND3', 'D2']
        elif self.MS_gasfracs.shape[1] == 4 and not deuterated:
            self.MS_labels = ['Ar', '$N_{2}$', '$NH_{3}$', '$H_{2}$']
            self.MS_species = ['Ar', 'N2', 'NH3', 'H2']
        else:
            pass
        return
    
    def get_log_data(self, CRD=False):
        if CRD:
            self.CRD = True
            self.log_data = pd.read_csv(self.log_fpath, sep='\t', 
        	                   usecols=[4, 6, 8, 9, 10, 11, 12, 23, 24],
        			   names=['Time', 'Pressure', 'MFM', 'MFC1_set', 'MFC1',
        			          'MFC2_set', 'MFC2', 'CRD', 'CRD_set'], skiprows=1)
        else:
            self.CRD = False
            self.log_data = pd.read_csv(self.log_fpath, sep='\t', 
                                        usecols=[4, 6, 8, 9, 10, 11, 12, 15, 16, 17],
                                        names=['Time', 'Pressure', 'MFM', 'MFC1_set', 'MFC1',
                                               'MFC2_set', 'MFC2', 'Main', 
                                               'Main_set', 'Aux'], skiprows=1)
    
    def plot_log(self, names, figsize=(10, 7), xlabel='Time / min',
                 ylabel=u'Temperature / \u00B0C', dpi=None):
        """Plot traces from log file
        
        Args:
            names: string or list of strings with the names of the traces to 
            be plotted.
            figsize (tup): (m, n) determines dimensions of figure
            xlabel (str): x-axis label
            ylabel (str): y-axis label
            dpi: resolution of the plot
        """
        if type(names) == str:
            names = [names]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        for n in names:
            ax.plot(self.log_data['Time'].values / 60, self.log_data[n].values,
                    label=n)
        if len(names) > 1:
            ax.legend()
        fig.tight_layout()
        return fig, ax
    
    def align_temperature_with_MS(self):
        indices = np.searchsorted(self.log_data['Time'].values / 60, 
                                  self.MS_fit_times)
        indices = np.where(indices < self.log_data.shape[0], indices, 
                           indices - 1)
        if self.CRD:
            self.T_MS = self.log_data['CRD'].values[indices]
        else:
            self.T_MS = self.log_data['Aux'].values[indices]
    
    def plot_MS_vs_T(self, figsize=(10, 7), xlabel=u'Temperature / \u00B0C', 
                     ylabel='Gas Fraction', legend=True, dpi=None):
        """Plot fitted MS traces versus temperature
        
        Args:
            figsize (tup): (m, n) determines dimensions of figure
            xlabel (str): x-axis label
            ylabel (str): y-axis label
            legend (bool): whether to include a legend or not
            dpi: resolution of the plot
        """
        self.align_temperature_with_MS()
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        for p, l in zip(self.MS_gasfracs.T, self.MS_labels):
            ax.plot(self.T_MS, p, label=l)
        if legend:
            ax.legend()
        fig.tight_layout()
        return fig, ax
    
    def get_temp_changes(self):
        """Return temperature change indices"""
        if self.CRD:
            Tchanges = self.log_data['CRD_set'].values[1:] - \
                       self.log_data['CRD_set'].values[:-1]
        else:
            Tchanges = self.log_data['Main_set'].values[1:] - \
                       self.log_data['Main_set'].values[:-1]
        Tc_indices = []
        for i, Tc in enumerate(Tchanges[:-1]):
            if Tc == 0 and Tchanges[i + 1] != 0:
                Tc_indices.append(i + 1)
        if self.CRD:
            self.Tc_indices = TC_Indices(Tc_indices, self.log_data, CRD=True)
        else:
            self.Tc_indices = TC_Indices(Tc_indices, self.log_data, CRD=False)
        
    def get_sigmoid_data(self, index_offset=150):
        """Return sigmoid data for all changes in temperature
        
        Args:
            index_offset: number of points to take averages over
        """
        if self.Tc_indices is None:
            self.get_temp_changes()
        times = self.log_data['Time'].values / 60.
        Tc_indices = self.Tc_indices.indices
        Tc_indices2 = self.Tc_indices.indices - index_offset
        Tc_ind_mspec = np.searchsorted(self.MS_fit_times, times[Tc_indices])
        Tc_ind_mspec2 = np.searchsorted(self.MS_fit_times, times[Tc_indices2])
        av_eq_mspec = np.row_stack([np.mean(self.MS_gasfracs[Tci2:Tci, :],
                                            axis=0) for Tci, Tci2 in
                                    zip(Tc_ind_mspec, Tc_ind_mspec2)])
        if self.CRD:
            T = self.log_data['CRD'].values
        else:
            T = self.log_data['Aux'].values
        av_eq_T = np.array([np.mean(T[Tci2:Tci]) for Tci, Tci2 in 
                            zip(Tc_indices, Tc_indices2)])
        av_eq_mspec_std = np.row_stack([np.mean(self.MS_gasfracs[Tci2:Tci, :],
                                            axis=0) for Tci, Tci2 in
                                    zip(Tc_ind_mspec, Tc_ind_mspec2)])
        av_eq_mspec[av_eq_mspec < 0] = 0
        conv = (1 - av_eq_mspec[:, 2]) / (1 + av_eq_mspec[:, 2])
        conv_unc = conv * av_eq_mspec[:, 2] / av_eq_mspec[:, 2]
        conv2 = 2 * av_eq_mspec[:, 1] / (1 - 2 * av_eq_mspec[:, 1])
        conv2_unc = conv2 * 2 * av_eq_mspec_std[:, 1] / av_eq_mspec[:, 1]
        self.av_eq_T = av_eq_T
        self.conv = conv
        self.conv_std = conv_unc
        self.conv2 = conv2
        self.conv2_unc = conv2_unc
    
    def plot_conv_only(self, figsize=(10, 7), dpi=None):
        """Plot traces from log file
        
        Args:
            figsize (tup): (m, n) determines dimensions of figure
            dpi: resolution of the plot
        """
        if use_conv2:
            C = self.conv2
        else:
            C = self.conv
        if C is None:
            raise ValueError("Need to have run get_sigmoid_data first")
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=u'Temperature / \u00B0C',
                             ylabel='Conversion', ylim=[0, 1])
        ax.scatter(self.av_eq_T, C, s=120)
        fig.tight_layout()
        return fig, ax
    
    def gomp(self, T, A, EA, R=8.3144598e-3):
        """Return conversion array for given T, A, EA"""
        a = A / R - EA / (R * T)
        if type(a) != type(np.array([])):
            a = np.array(a)
        a[a > 10] = 10
        a[a < -10] = -10
        return 1 - np.exp(-np.exp(a))
    
    def gomp_Te(self, T, Te, EA, R=8.3144598e-3):
        A = float(EA) / float(Te)
        return self.gomp(T, A, EA, R)
    
    def gomp_Te_fadd(self, T, params, R=8.3144598e-3):
        """Adding two fractions of Gompertz functions (a la Bill)"""
        Te1, EA1, Te2, EA2, f = params
        return f * self.gomp_Te(T, Te1, EA1) + (1 - f) * self.gomp_Te(T, Te2, EA2)
    
    def fit_single_Te(self, guesses, use_conv2=False):
        """Return best Te and Ea parameters for function to data
    
        Args:
            guesses: list of Te, EA guesses
            use_conv2 (bool): whether to use conversion data from N2 fraction
            or keep using the NH3 fraction calculation
        """
        T = self.av_eq_T + 273.15
        if use_conv2:
            C = self.conv2
        else:
            C = self.conv
        C = 1 - (1 - C) / (1 + C)
        self.Tfit = np.linspace(T.min(), T.max(), 1000)
        R = 8.3144598e-3 # in kJ mol^-1 K^-1
        def residuals(guesses, T, C, R):
            Te, EA = guesses
            Ccalc = self.gomp_Te(T, Te, EA)
            err = C - Ccalc
            return err
        params = opt.leastsq(residuals, guesses, args=(T, C, R), 
                             full_output=True)
        if type(params[1]) == type(None):
            cov_p = np.ones((2, 2)) * np.inf
        else:
            s_sq = (params[2]['fvec']**2).sum() / (len(params[2]['fvec']) -\
                                                   len(params[0]))
            cov_p = params[1] * s_sq
#        s_sq_sig = ((params[2]['fvec'] / 0.01)**2).sum() / \
#                   (len(params[2]['fvec']) - len(params[0]))
        p = params[0]
        self.conv_single_std = np.diagonal(cov_p)**0.5
        self.conv_single_params = p
        self.conv_single_cov = params[1]
        self.conv_single_fit = self.gomp_Te(self.Tfit, p[0], p[1])
        self.conv_single_corr = self.cov2corr(params[1])
    
    def fit_double_Te(self, guesses, use_conv2=False):
        """Return best Te1, EA1, Te2, EA2, f parameters for function to data
    
        Args:
            guesses: list of A1, EA1, A2, EA2 guesses
            use_conv2 (bool): whether to use conversion data from N2 fraction
            or keep using the NH3 fraction calculation
        """
        T = self.av_eq_T + 273.15
        if use_conv2:
            C = self.conv2
        else:
            C = self.conv
        C = 1 - (1 - C) / (1 + C)
        self.Tfit = np.linspace(T.min(), T.max(), 1000)
        def residuals(guesses, T, C):
            Ccalc = self.gomp_Te_fadd(T, guesses)
            err = C - Ccalc
            return err
        params = opt.leastsq(residuals, guesses, args=(T, C), full_output=True)
        if type(params[1]) == type(None):
            cov_p = np.ones((2, 2)) * np.inf
        else:
            s_sq = (params[2]['fvec']**2).sum() / (len(params[2]['fvec']) -\
                                                   len(params[0]))
            cov_p = params[1] * s_sq
        p = params[0]
        self.conv_double_std = np.diagonal(cov_p)**0.5
        self.conv_double_params = p
        self.conv_double_cov = params[1]
        self.conv_double_fit = self.gomp_Te_fadd(self.Tfit, p)
        if params[1] is not None:
            self.conv_double_corr = self.cov2corr(params[1])
    
    def cov2corr(self, A):
        d = np.sqrt(np.matrix(A).diagonal())
        res = (A.T / d).T / d
        return res
    
    def plot_conv_guesses(self, guesses, figsize=(10, 7), dpi=None,
                          use_conv2=False):
        """Plot traces from log file
        
        Args:
            guesses (list): list of parameter guesses for either single or
            double fit.
            figsize (tup): (m, n) determines dimensions of figure
            dpi: resolution of the plot
            use_conv2 (bool): whether to use conversion data from N2 fraction
            or keep using the NH3 fraction calculation
        """
        if use_conv2:
            C = self.conv2
        else:
            C = self.conv
        if self.conv is None or self.conv2 is None:
            raise ValueError("Need to have run get_sigmoid_data first")
        T = self.av_eq_T + 273.15
        if self.Tfit is None:
            self.Tfit = np.linspace(T.min(), T.max(), 1000)
        if len(guesses) == 2:
            Ccalc = self.gomp_Te(self.Tfit, guesses[0], guesses[1]) 
        elif len(guesses) == 5:
            Ccalc = self.gomp_Te_fadd(self.Tfit, guesses)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=u'Temperature / \u00B0C',
                             ylabel='Conversion', ylim=[0, 1])
        ax.scatter(self.av_eq_T, C, s=120)
        ax.plot(self.Tfit - 273.15, Ccalc / (2 - Ccalc))
        fig.tight_layout()
        return fig, ax
    
    
    def plot_conv_fit(self, figsize=(10, 7), dpi=None, use_conv2=False):
        """Plot conversion with fits
        
        Args:
            figsize (tup): (m, n) determines dimensions of figure
            dpi: resolution of the plot
            use_conv2 (bool): whether to use conversion data from N2 fraction
            or keep using the NH3 fraction calculation
        """
        fits = []
        labels = []
        if use_conv2:
            C = self.conv2
        else:
            C = self.conv
        if self.conv_single_fit is not None:
            fits.append(self.conv_single_fit / (2 - self.conv_single_fit))
            labels.append('single')
        if self.conv_double_fit is not None:
            fits.append(self.conv_double_fit / (2 - self.conv_double_fit))
            labels.append('double')
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.scatter(self.av_eq_T, C, s=120)
        for f, l in zip(fits, labels):
            ax.plot(self.Tfit - 273.15, f, label=l)
        ax.legend()
        ax.set_xlabel(u'Temperature / \u00B0C')
        ax.set_ylabel('Conversion')
        ax.tick_params(top=False, right=False)
        ax.set_ylim([0, 1])
        fig.tight_layout()
        return fig, ax
    
    def save_sigfit(self, save_fname, fit_type='single'):
        sf = self.fpath + save_fname
        if fit_type == 'single':
            f = self.conv_single_fit
        elif fit_type == 'double':
            f = self.conv_double_fit
        else:
            raise ValueError("fit_type should be 'single' or 'double'")
        fits = np.column_stack((self.Tfit - 273.15, f / (2 - f)))
        np.savetxt(sf, fits)

class TC_Indices:
    def __init__(self, indices, log_data, CRD=False):
        self.indices = np.array(indices)
        self.old_indices = np.array(indices)
        self.CRD = CRD
        self.log_data = log_data
    
    def remove(self, index_list):
        """Removes each TC_index from its position in index_list"""
        keep_is = ([a for a in range(len(self.indices)) if a not in index_list])
        self.indices = self.indices[keep_is]
    
    def reset(self):
        self.indices = np.copy(self.old_indices)
    
    def plot_changes(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, xlabel='Time / min', 
                             ylabel=u'Temperature / \u00B0C')
        t = self.log_data['Time'].values / 3600
        if self.CRD:
            T = self.log_data['CRD'].values
        else:
            T = self.log_data['Aux'].values
        ax.plot(t, T)
        for n, i in enumerate(self.indices):
            ax.axvline(t[i], ls='dashed', color='k')
            ax.text(t[i], T[i], str(n), ha='right', va='bottom', color='r')
        fig.tight_layout()
        return fig, ax
    

def extract_single_fit(Tfits, fits):
    """Return single fit, with blank contribution removed
    
    Args:
        Tfits (list): list of x data arrays (temperature - doesn't matter 
        which units).
        fits (list): list of: (i) 1 - fNH3 data fit for catalyst and (ii)
        1 - fNH3 data fit for blank reactor.
    """
    Tf0, Tf1 = Tfits
    blank_int = np.interp(Tf0, Tf1, fits[1])
    res = (fits[0] - blank_int) / (blank_int * (fits[0] - 2) + 1)
    return res

class Calibration_Experiment:
    def __init__(self, fpath, nh3_h2_n2_ratios = [0.0996, 0.6753, 0.2251],
                 gcfs={'ar' : 1.4047 / 0.7807, 'nh3': 1., 'n2': 1. / 0.7807,
                       'h2' : 1.0038 / 0.7807}):
        self.fpath = fpath
        self.get_fnames()
        self.MS_times = None
        self.MS_amus = None
        self.MS_ps = None
        self.MS_fracs = None
        self.nh3_h2_n2_ratios = nh3_h2_n2_ratios
        self.gcfs = gcfs
        self.gas_mixes = {}
    
    def get_fnames(self):
        import os
        fnames = os.listdir(self.fpath)
        csv_fnames = [fn for fn in fnames if fn[-4:] == '.csv']
        log_fnames = [fn for fn in fnames if fn[-4:] == '.log']
        if len(csv_fnames) == 0:
            ps = ('Could not find a .csv file in the named filepath\n'
                  'Please supply csv_fpath manually')
            print(ps)
        elif len(csv_fnames) == 1:
            self.csv_fpath = self.fpath + csv_fnames[0]
            print(f'csv_fname set as "{csv_fnames[0]}"')
        else:
            self.csv_fpath = self.fpath + csv_fnames[0]
            ps = ('Multiple .csv files found in the named filepath\n'
                  'csv_fname set as "{csv_fnames[0]}"')
        if len(log_fnames) == 0:
            ps = ('Could not find a .log file in the named filepath\n'
                  'Please supply log_fpath manually')
            print(ps)
        elif len(log_fnames) == 1:
            self.log_fpath = self.fpath + csv_fnames[0]
            print(f'log_fname set as "{log_fnames[0]}"')
        else:
            self.csv_fname = self.fpath + csv_fnames[0]
            ps = ('Multiple .csv files found in the named filepath\n'
                  'log_fname set as "{log_fnames[0]}"')
        return
    
    def extract_MS_data(self, sums=None):
        """Extract MS data into times, amus and fracs
        
        Args:
            sums (int): number of histogram instances to sum (take mean) over
        """
        csv = pd.read_csv(self.csv_fpath, sep=',', header=29, 
                                   usecols=[0, 2, 3, 4])
        cyc_len = int(csv['mass amu'].max())
        cyc_num = int(csv.shape[0] / cyc_len)
        if sums:
            times = np.zeros((cyc_len, cyc_num // sums))
            amus = np.zeros((cyc_len, cyc_num // sums))
            ps = np.zeros((cyc_len, cyc_num // sums))
            t = csv['ms'].values
            p = csv['Faraday torr'].values
            for n in range(cyc_num // sums):
                times[:, n] = np.mean(t[int(cyc_len * n * sums):\
                    int(cyc_len * (n + 1) * sums)].reshape(sums, cyc_len),\
                                     axis=0) / 6e4
                amus[:, n] = csv['mass amu'][int(cyc_len * n * sums):
                                             int(cyc_len * (n * sums + 1))]
                ps[:, n] = np.mean(p[int(cyc_len * n * sums):\
                    int(cyc_len * (n + 1) * sums)].reshape(sums, cyc_len),\
                                  axis=0)
        else:
            times = np.zeros((cyc_len, cyc_num))
            amus = np.zeros((cyc_len, cyc_num))
            ps = np.zeros((cyc_len, cyc_num))
            csv_vals = csv.values
            for n in range(cyc_num):
                times[:, n] = csv_vals[cyc_len * n:cyc_len * (n + 1), 1] / 6e4
                amus[:, n] = csv_vals[cyc_len * n:cyc_len * (n + 1), 2]
                ps[:, n] = csv_vals[cyc_len * n:cyc_len * (n + 1), 3]
        tot_ps = np.sum(ps, axis=0)
        tot_ps = np.meshgrid(tot_ps, np.zeros(ps.shape[0]))[0]
        self.MS_times = times
        self.MS_amus = amus
        self.MS_ps = ps
        self.MS_fracs = ps / tot_ps
    
    def MS_contour(self, colour_num=50, figsize=(10, 8), xlabel='Time / min',
                   ylabel='m/z / a.m.u.', zlabel='Gas Fraction', dpi=None,
                   zscale='linear', t_range=None):
        """Return filled contour plot of raw MS gas fraction data over time
        
        Args:
            colour_num (int): number of colours to use on z scale
            figsize (tuple): (m, n) size of figure in inches
            xlabel (str): label for x axis
            ylabel (str): label for y axis
            zlabel (str): label for z axis
            dpi (int): image resolution
            zscale (str): can take values of 'linear', 'sqrt' or 'log', where
            the log is natural log.
            t_range (list): list of first and last time values to plot over
        Returns:
            fig: plt.figure instance
            ax: plt.axis instance
            cbar: colourbar instance
        """
        t, a, f = self.MS_times, self.MS_amus, self.MS_fracs
        if t_range is not None:
            ti0, ti1 = np.searchsorted(t[0], t_range)
            t, a, f = t[:, ti0:ti1], a[:, ti0:ti1], f[:, ti0:ti1]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
        ax.tick_params(top=False, right=False)
        if zscale == 'linear':
            im = ax.contourf(t, a, f, colour_num)
        elif zscale == 'sqrt':
            im = ax.contourf(t, a, np.sqrt(np.clip(f, 1e-5, 1)), colour_num)
            zlabel = f"$\sqrt{{{zlabel}}}$"
        elif zscale == 'log':
            im = ax.contourf(t, a, np.log(np.clip(f, 1e-5, 1)), colour_num)
            zlabel = f"log({zlabel})"
        else:
            raise ValueError("zscale must take value of 'linear', 'sqrt' or 'log'")
        cbar = plt.colorbar(im)
        cbar.set_label(zlabel, rotation=270, labelpad=20)
        fig.tight_layout()
        return fig, ax, cbar
    
    def plot_MS_traces(self, amus=[2, 17, 28, 40], t_range=None,
                       figsize=(10, 7)):
        """Plot individual m/z traces"""
        if t_range is None:
            t_range = [0, self.MS_times.shape[1]]
        ti0, ti1 = t_range
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, xlabel='Time / min', ylabel='m/z fraction')
        for amu in amus:
            i = np.searchsorted(self.MS_amus[:, 0], amu)
            ax.plot(self.MS_times[0, ti0:ti1], self.MS_fracs[i, ti0:ti1], 
                    label=f"m/z {amu}")
        ax.legend()
        fig.tight_layout()
        return fig, ax
    
    def update_mixes(self, s, tvals, flows=None, cutoff=None):
        """Assign times where gases/gas mixtures are at equilibrium
        
        Args:
            s: string---can be 'ar', 'nh3', 'n2', 'arnh3', 'arn2', 'nh3h2n2'
            tvals: beginning and end times of equilibrium
            flows: list of flows in order (optional, if non-extant, then flows
            assumed to be the same)
            cutoff: cutoff fraction above which to consider things relevant
        """
        tis = np.searchsorted(self.MS_times[0], tvals)
        if len(s) < 4:
            self.gas_mixes.update({s.lower() : Mixture(tis, self.MS_fracs,
                                   flows=flows, cutoff=cutoff)})
        elif 'ar' in s.lower() and 'nh3' in s.lower():
            self.gas_mixes.update({'arnh3' : Mixture(tis, self.MS_fracs,
                                                flows=flows, cutoff=cutoff)})
        elif 'ar' in s.lower() and 'n2' in s.lower():
            self.gas_mixes.update({'arn2' : Mixture(tis, self.MS_fracs,
                                                flows=flows, cutoff=cutoff)})
        elif 'nh3' in s.lower() and 'n2' in s.lower() and 'h2' in s.lower():
            self.gas_mixes.update({'nh3h2n2' : Mixture(tis, self.MS_fracs,
                                                flows=flows, cutoff=cutoff)})
    
    def get_Ifactor_from_Ar_mix(self, mixture, ar_rats, gas_rats):
        """Return ionization factor for gas from mixture with argon
        
        Args:
            mixture (str): key for self.gas_mixes
            ar_rats (tuple): amus and fractions for pure argon
            gas_rats (tuple): amus and fractions for pure other gas
        """
        if 'ar' not in mixture:
            raise ValueError("Can't get ionization factor without Ar")
        gas = mixture[2:]
        if gas not in self.gas_mixes.keys():
            raise KeyError(f"Can't get ionization factor without pure {gas}")
        mix = self.gas_mixes[mixture]
        amus, fracs = mix.get_fracs_above_cutoff()
        ar_fs = np.zeros(40)
        ar_fs[ar_rats[0]-1] = ar_rats[1]
        gas_fs = np.zeros(40)
        gas_fs[gas_rats[0]-1] = gas_rats[1]
        mix_fs = np.zeros(40)
        mix_fs[amus - 1] = fracs
        def residuals(guesses, mix_fs, ar_fs, gas_fs):
            a, b = guesses
            return np.abs(mix_fs - b * (a * ar_fs + (1 - a) * gas_fs))
        from scipy.optimize import leastsq
        p = leastsq(residuals, np.array([0.5, 1.]), args=(mix_fs, ar_fs,
                    gas_fs))[0]
        ar_flow = mix.flows[0] * self.gcfs['ar']
        gas_flow = mix.flows[1] * self.gcfs[gas]
        actual_a = ar_flow / (ar_flow + gas_flow)
        a, b = p[0], actual_a
        return (b / (1 - b)) / (a / (1 - a))
    
    def calculate_fragmentations_and_Ifactors(self):
        for s in ['ar', 'n2', 'nh3', 'arnh3', 'arn2', 'nh3h2n2']:
            if s not in self.gas_mixes.keys():
                raise KeyError(f'{s} not found in gas_mixes')
        ar_rats = self.gas_mixes['ar'].get_fracs_above_cutoff()
        ar_I = 1.
        nh3_rats = self.gas_mixes['nh3'].get_fracs_above_cutoff()
        nh3_I = self.get_Ifactor_from_Ar_mix('arnh3', ar_rats, nh3_rats)
        n2_rats = self.gas_mixes['n2'].get_fracs_above_cutoff()
        n2_I = self.get_Ifactor_from_Ar_mix('arn2', ar_rats, n2_rats)
        #next to do is work out h2 fragmentation ratios from the
        #NH3/N2/H2 mixture (use unique NH3/N2 to calculate that mix and then
        #H2 is whatever's left over in the 1-3 range)
        amus, fracs = self.gas_mixes['nh3h2n2'].get_fracs_above_cutoff()
        nh3_fs = np.zeros(40)
        nh3_fs[nh3_rats[0]-1] = nh3_rats[1]
        n2_fs = np.zeros(40)
        n2_fs[n2_rats[0]-1] = n2_rats[1]
        mix_fs = np.zeros(40)
        mix_fs[amus - 1] = fracs
        def residuals(guesses, mix_fs, nh3_fs, n2_fs):
            a, b = guesses
            return mix_fs[3:] - (a * nh3_fs[3:] + b * n2_fs[3:])
        from scipy.optimize import leastsq
        p = leastsq(residuals, np.array([0.3, 0.7]), args=(mix_fs, nh3_fs,
                    n2_fs))[0]
        h2_amus = amus[amus <= 3]
        h2_fracs = (mix_fs - nh3_fs * p[0])[:3]
        h2_f = np.sum(h2_fracs)
        h2_fracs /= h2_f
        h2_I = h2_f * n2_I / ((self.nh3_h2_n2_ratios[1] / \
                               self.nh3_h2_n2_ratios[2]) * p[1])
        h2_rats = (h2_amus, h2_fracs)
        self.Ifactors = IonizationFactors(I_ar=ar_I, I_nh3=nh3_I, I_h2=h2_I,
                                          I_n2=n2_I)
        self.frag_rats = FragmentationRatios(ar_rats=ar_rats, nh3_rats=nh3_rats,
                                             n2_rats=n2_rats, h2_rats=h2_rats)
        return

class Mixture:
    def __init__(self, tis, MS_fracs, flows=None, cutoff=None):
        if cutoff:
            self.cutoff = cutoff
        else:
            self.cutoff = 0.0028
        self.tis = tis
        self.get_average_fracs(MS_fracs)
        if flows:
            self.flows = flows
        else:
            self.flows = [1., 1.]
        print(self)
    
    def get_average_fracs(self, MS_fracs):
        ti0, ti1 = self.tis
        self.av_fracs = MS_fracs[:, ti0:ti1].mean(axis=1)
        self.std_fracs = MS_fracs[:, ti0:ti1].std(axis=1)
    
    def __str__(self):
        inds = self.av_fracs > self.cutoff
        fracs = self.av_fracs[inds]
        amus = (np.arange(self.av_fracs.shape[0]) + 1)[inds]
        s = f'Mixture (all fracs > {self.cutoff}):\n\tm/z\tfrac\n'
        for amu, f in zip(amus, fracs):
            s += f'\t{amu}\t{f:.3f}\n'
        return s
    
    def get_fracs_above_cutoff(self):
        inds = self.av_fracs > self.cutoff
        fracs = self.av_fracs[inds]
        fracs /= np.sum(fracs)
        amus = (np.arange(self.av_fracs.shape[0]) + 1)[inds]
        return amus, fracs
    
    def plot_bar(self, figsize=(10, 7), use_cutoff=True):
        if use_cutoff:
            amus, fracs = self.get_fracs_above_cutoff()
        else:
            amus = np.arange(1, 41)[self.av_fracs > 0]
            fracs = self.av_fracs[self.av_fracs > 0]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, xlabel='m/z', 
                             ylabel='Fragmentation fraction')
        ax.bar(amus, fracs)
        fig.tight_layout()
        return fig, ax