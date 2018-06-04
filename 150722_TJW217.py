import numpy as np
from usefuldata150626 import *
from mscalib141008 import *
from mscoeffs150706 import *
from scipy.optimize import leastsq
import pandas as pd

def extract_mshist_pd(csv, bins=None, sums=None):
    """Return times, amus and pressures for csv file array"""
    cyc_len = csv['mass amu'].max()
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

def fit_all_halfstatar(M, sfs, ex_fs, Is, fit_array=True, guesses=None):
    """Return the best fit (LM) for MS data assuming statistical am-species.

    Args:
        M: array of measured MS fractions (normalized)
        sfs: list of sensitivity factors for fragmentations of molecules in
        molecular weight order.
        ex_fs: list of factors for probability of adding a hydrogen/
        deuterium onto ammonia(-d) or onto H2/D2.
        Is: ionization factors in molecular weight order
        fit_array (bool): determines whether to return fitted array or not
        guesses: list of guesses for fraction of N2, fraction ammonia(-d),
        fractions of H2, HD, D2, fraction of ammonia which is deuterated.
    Returns:
    """
    h2_s, hd_s, d2_s, nh3_s, ndh2_s, nd2h_s, nd3_s, n2_s, ar_s = sfs
    ex_H, ex_D, ex_H2, ex_D2 = ex_fs
    I_h2, I_hd, I_d2, I_nh3, I_ndh2, I_nd2h, I_nd3, I_n2, I_ar = Is
    #work out initial guesses
    if type(guesses) == type(None):
        g_ar = M[15] / ar_s[15]
        if g_ar > 1:
            g_ar = 1
        elif g_ar < 0:
            g_ar = 0
        g_n2 = M[14] / n2_s[14]
        if g_n2 > 1:
            g_n2 = 1
        elif g_n2 < 0:
            g_n2 = 0
        g_am = np.sum(M[5:14]) / np.sum(M)
        if M[8] + M[11] <= 0:
            f_Dam = 0
        else:
            f_Dam = M[11] / (M[8] + M[11])
        g_hyd = 1 - g_n2 - g_am - g_ar
        if M[0] + M[1] + M[2] <= 0:
            f_Dhyd = 0
        else:
            f_Dhyd = (M[2] + 0.5 * M[1]) / (M[0] + M[1] + M[2])
        g_h2 = g_hyd * (1 - f_Dhyd)**2
        g_hd = g_hyd * 2 * (1 - f_Dhyd) * f_Dhyd
        g_d2 = g_hyd * f_Dhyd**2
        guesses = [g_ar, g_n2, g_am, g_h2, g_hd, g_d2, f_Dam]
    else:
        g_ar, g_n2, g_am, g_h2, g_hd, g_d2, f_Dam = guesses
    if fit_array:
        r_nh3 = (1 - f_Dam)**3 * I_nh3
        r_ndh2 = 3 * f_Dam * (1 - f_Dam)**2 * I_ndh2
        r_nd2h = 3 * f_Dam**2 * (1 - f_Dam) * I_nd2h
        r_nd3 = f_Dam**3 * I_nd3
        g_nh3, g_ndh2, g_nd2h, g_nd3 = \
                [g_am * r for r in normalize_coeffs([r_nh3, r_ndh2, r_nd2h,
                                                     r_nd3])]
        #now adjust for hydrogen/deuterium ratio
        frac_D = g_am * f_Dam + g_d2 + 0.5 * g_hd
        frac_H = g_am * (1 - f_Dam) + g_h2 + 0.5 * g_hd
        nh3_s1 = nh3_s[:]
        nh3_s1[9] = nh3_s[8] * ex_H * frac_H
        nh3_s1[10] = nh3_s[8] * ex_D * frac_D
        nh3_s1 = np.array(normalize_coeffs(nh3_s1))
        ndh2_s1 = ndh2_s[:]
        ndh2_s1[10] = ndh2_s[9] * ex_H * frac_H
        ndh2_s1[11] = ndh2_s[9] * ex_D * frac_D
        ndh2_s1 = np.array(normalize_coeffs(ndh2_s1))
        nd2h_s1 = nd2h_s[:]
        nd2h_s1[11] = nd2h_s[10] * ex_H * frac_H
        nd2h_s1[12] = nd2h_s[10] * ex_D * frac_D
        nd2h_s1 = np.array(normalize_coeffs(nd2h_s1))
        nd3_s1 = nd3_s[:]
        nd3_s1[12] = nd3_s[11] * ex_H * frac_H
        nd3_s1[13] = nd3_s[11] * ex_D * frac_D
        nd3_s1 = np.array(normalize_coeffs(nd3_s1))
        n2_s1 = np.array(n2_s[:])
        ar_s1 = np.array(ar_s[:])
        h2_s1 = h2_s[:]
        h2_s1[1] = h2_s[0] * ex_H2 * frac_H
        h2_s1[2] = h2_s[0] * ex_D2 * frac_D
        h2_s1 = np.array(normalize_coeffs(h2_s1))
        hd_s1 = hd_s[:]
        hd_s1[2] = hd_s[1] * ex_H2 * frac_H
        hd_s1[3] = hd_s[1] * ex_D2 * frac_D
        hd_s1 = np.array(normalize_coeffs(hd_s1))
        d2_s1 = d2_s[:]
        d2_s1[3] = d2_s[2] * ex_H2 * frac_H
        d2_s1[4] = d2_s[2] * ex_D2 * frac_D
        d2_s1 = np.array(normalize_coeffs(d2_s1))
        guess_M = g_n2 * n2_s1 + g_nd3 * nd3_s1 + g_nd2h * nd2h_s1 +\
                  g_ndh2 * ndh2_s1 + g_nh3 * nh3_s1 + g_d2 * d2_s1 +\
                  g_hd * hd_s1 + g_h2 * h2_s1 + g_ar * ar_s1
    def residuals(guesses, sfs, ex_fs, Is, M):
        h2_s, hd_s, d2_s, nh3_s, ndh2_s, nd2h_s, nd3_s, n2_s, ar_s = sfs
        ex_H, ex_D, ex_H2, ex_D2 = ex_fs
        I_h2, I_hd, I_d2, I_nh3, I_ndh2, I_nd2h, I_nd3, I_n2, I_ar = Is
        g_ar, g_n2, g_am, g_h2, g_hd, g_d2, f_Dam = guesses
        g_ar, g_n2, g_am, g_h2, g_hd, g_d2 = \
                normalize_coeffs([g_ar, g_n2, g_am, g_h2, g_hd, g_d2])
        r_nh3 = (1 - f_Dam)**3 * I_nh3
        r_ndh2 = 3 * f_Dam * (1 - f_Dam)**2 * I_ndh2
        r_nd2h = 3 * f_Dam**2 * (1 - f_Dam) * I_nd2h
        r_nd3 = f_Dam**3 * I_nd3
        g_nh3, g_ndh2, g_nd2h, g_nd3 = \
                [g_am * r for r in normalize_coeffs([r_nh3, r_ndh2, r_nd2h,
                                                     r_nd3])]
        frac_D = g_am * f_Dam + g_d2 + 0.5 * g_hd
        frac_H = g_am * (1 - f_Dam) + g_h2 + 0.5 * g_hd
        nh3_s1 = nh3_s[:]
        nh3_s1[9] = nh3_s[8] * ex_H * frac_H
        nh3_s1[10] = nh3_s[8] * ex_D * frac_D
        nh3_s1 = np.array(normalize_coeffs(nh3_s1))
        ndh2_s1 = ndh2_s[:]
        ndh2_s1[10] = ndh2_s[9] * ex_H * frac_H
        ndh2_s1[11] = ndh2_s[9] * ex_D * frac_D
        ndh2_s1 = np.array(normalize_coeffs(ndh2_s1))
        nd2h_s1 = nd2h_s[:]
        nd2h_s1[11] = nd2h_s[10] * ex_H * frac_H
        nd2h_s1[12] = nd2h_s[10] * ex_D * frac_D
        nd2h_s1 = np.array(normalize_coeffs(nd2h_s1))
        nd3_s1 = nd3_s[:]
        nd3_s1[12] = nd3_s[11] * ex_H * frac_H
        nd3_s1[13] = nd3_s[11] * ex_D * frac_D
        nd3_s1 = np.array(normalize_coeffs(nd3_s1))
        n2_s1 = np.array(n2_s[:])
        ar_s1 = np.array(ar_s[:])
        h2_s1 = h2_s[:]
        h2_s1[1] = h2_s[0] * ex_H2 * frac_H
        h2_s1[2] = h2_s[0] * ex_D2 * frac_D
        h2_s1 = np.array(normalize_coeffs(h2_s1))
        hd_s1 = hd_s[:]
        hd_s1[2] = hd_s[1] * ex_H2 * frac_H
        hd_s1[3] = hd_s[1] * ex_D2 * frac_D
        hd_s1 = np.array(normalize_coeffs(hd_s1))
        d2_s1 = d2_s[:]
        d2_s1[3] = d2_s[2] * ex_H2 * frac_H
        d2_s1[4] = d2_s[2] * ex_D2 * frac_D
        d2_s1 = np.array(normalize_coeffs(d2_s1))
        M_calc = g_n2 * n2_s1 + g_nd3 * nd3_s1 + g_nd2h * nd2h_s1 +\
                 g_ndh2 * ndh2_s1 + g_nh3 * nh3_s1 + g_d2 * d2_s1 +\
                 g_hd * hd_s1 + g_h2 * h2_s1 + g_ar * ar_s1
        err = np.abs(M - M_calc)
        return err
    params = leastsq(residuals, guesses, args=(sfs, ex_fs, Is, M))
    p = list(params[0])
    #Now adjust for ionization factors
    I_am = (1 - p[4])**3 * I_nh3 + 3 * p[4] * (1 - p[4])**2 * I_ndh2 +\
           3 * p[4]**2 * (1 - p[4]) * I_nd2h + p[4]**3 * I_nd3
    p = [p[0] / I_ar, p[1] / I_n2, p[2] / I_am, p[3] / I_h2, p[4] / I_hd,
         p[5] / I_d2] + [p[6]]
    p = normalize_coeffs(p[:6]) + [p[6]]
    if fit_array:
        g_ar, g_n2, g_am, g_h2, g_hd, g_d2, f_Dam = p
        r_nh3 = (1 - f_Dam)**3 * I_nh3
        r_ndh2 = 3 * f_Dam * (1 - f_Dam)**2 * I_ndh2
        r_nd2h = 3 * f_Dam**2 * (1 - f_Dam) * I_nd2h
        r_nd3 = f_Dam**3 * I_nd3
        g_nh3, g_ndh2, g_nd2h, g_nd3 = \
                [g_am * r for r in normalize_coeffs([r_nh3, r_ndh2, r_nd2h,
                                                     r_nd3])]
        #now adjust for hydrogen/deuterium ratio
        frac_D = g_am * f_Dam + g_d2 + 0.5 * g_hd
        frac_H = g_am * (1 - f_Dam) + g_h2 + 0.5 * g_hd
        nh3_s1 = nh3_s[:]
        nh3_s1[9] = nh3_s[8] * ex_H * frac_H
        nh3_s1[10] = nh3_s[8] * ex_D * frac_D
        nh3_s1 = np.array(normalize_coeffs(nh3_s1))
        ndh2_s1 = ndh2_s[:]
        ndh2_s1[10] = ndh2_s[9] * ex_H * frac_H
        ndh2_s1[11] = ndh2_s[9] * ex_D * frac_D
        ndh2_s1 = np.array(normalize_coeffs(ndh2_s1))
        nd2h_s1 = nd2h_s[:]
        nd2h_s1[11] = nd2h_s[10] * ex_H * frac_H
        nd2h_s1[12] = nd2h_s[10] * ex_D * frac_D
        nd2h_s1 = np.array(normalize_coeffs(nd2h_s1))
        nd3_s1 = nd3_s[:]
        nd3_s1[12] = nd3_s[11] * ex_H * frac_H
        nd3_s1[13] = nd3_s[11] * ex_D * frac_D
        nd3_s1 = np.array(normalize_coeffs(nd3_s1))
        n2_s1 = np.array(n2_s[:])
        ar_s1 = np.array(ar_s[:])
        h2_s1 = h2_s[:]
        h2_s1[1] = h2_s[0] * ex_H2 * frac_H
        h2_s1[2] = h2_s[0] * ex_D2 * frac_D
        h2_s1 = np.array(normalize_coeffs(h2_s1))
        hd_s1 = hd_s[:]
        hd_s1[2] = hd_s[1] * ex_H2 * frac_H
        hd_s1[3] = hd_s[1] * ex_D2 * frac_D
        hd_s1 = np.array(normalize_coeffs(hd_s1))
        d2_s1 = d2_s[:]
        d2_s1[3] = d2_s[2] * ex_H2 * frac_H
        d2_s1[4] = d2_s[2] * ex_D2 * frac_D
        d2_s1 = np.array(normalize_coeffs(d2_s1))
        fit_M = g_n2 * n2_s1 + g_nd3 * nd3_s1 + g_nd2h * nd2h_s1 +\
                g_ndh2 * ndh2_s1 + g_nh3 * nh3_s1 + g_d2 * d2_s1 +\
                g_hd * hd_s1 + g_h2 * h2_s1 + g_ar * ar_s1
        return guesses, p, guess_M, fit_M
    return guesses, p

fname = "150709_TJW217_HF_R4N_NaND2+ND3"
fpath = "/home/tomwood/Documents/Data/" + fname + "/" + fname + ".csv"

"""
csv = pd.read_csv(fpath, sep=',', header=29, usecols=[0, 2, 3, 4])

times, amus, ps = extract_mshist_pd(csv, sums=20)
tot_ps = np.sum(ps, axis=0)
fracs = np.zeros(ps.shape)
for i, col in enumerate(ps[0, :]):
    fracs[:, i] = ps[:, i] / tot_ps[i]

"""
#get temperatures and flows from log file
log = pd.read_csv(fpath[:-3] + 'log', sep='\t',
                  usecols=[4, 8, 9, 10, 11, 12, 15, 16, 17],
                  names=['Time', 'MFM', 'MFC1_set', 'MFC1', 'MFC2_set',
                         'MFC2', 'Main', 'Main_set', 'Aux'], skiprows=1)
ms = pd.read_csv('TJW217_MSfits.csv', names=['t', 'Ar', 'N2', 'ND3', 'D2'])
#work out Temp setpoint changes
T_changes = log['Main_set'].values[1:] - log['Main_set'].values[:-1]
T_czero = np.logical_not(T_changes)
T_cpos = np.logical_not(T_czero)
t_diffs = log['Time'].values[1:][np.logical_and(T_cpos[1:], T_czero[:-1])]
ms_t_changes = [np.abs(ms['t'].values -  t).argmin() for t in 
                t_diffs[1:] / 60]
log_t_changes = [np.abs(log['Time'].values - t).argmin() for t in
                 t_diffs[1:]]

#compute averages for relevant values
av_t = 5
log_t_av = [np.abs(log['Time'].values - t + av_t * 60).argmin() for t in 
            t_diffs[1:]]
log_avs = np.zeros((len(log_t_changes), 3))
log_sds = np.zeros((len(log_t_changes), 3))
for i, ch in enumerate(log_t_changes):
    log_avs[i, :] = log[['MFM', 'MFC2', 'Aux']][log_t_av[i]:ch].mean()
    log_sds[i, :] = log[['MFM', 'MFC2', 'Aux']][log_t_av[i]:ch].std()
ms_t_av = [np.abs(ms['t'].values - t + av_t).argmin() for t in 
           t_diffs[1:] / 60]
ms_avs = np.zeros((len(ms_t_changes), 4))
ms_sds = np.zeros((len(ms_t_changes), 4))
for i, ch in enumerate(ms_t_changes):
    ms_avs[i, :] = ms[['Ar', 'N2', 'ND3', 'D2']][ms_t_av[i]:ch].mean()
    ms_sds[i, :] = ms[['Ar', 'N2', 'ND3', 'D2']][ms_t_av[i]:ch].std()
labels = ['f$_{Ar}$', 'f$_{N_{2}}$', 'f$_{ND_{3}}$', 'f$_{D_{2}}$']
"""
#try some more temporary sensitivity factors excluding m/z
ex_fs = [ex_H, ex_D]

#the following is for using fit_all_justdar
mzs2 = [2, 3, 4, 6, 14, 16, 18, 20, 28, 36, 40]
n2_sfs2 = temp_fs(n2_sfs, mzs2)
I2_n2 = I_n2 * (1 - n2_sfs[2][1])
#redo ar_sfs based on first 180 minutes:
ar_t = [np.searchsorted(times[0, :], i) for i in [10, 35]]
ar_sum = np.mean(fracs[:, ar_t[0]:ar_t[1]], axis=1)
ar_conts = get_conts(ar_sum)
ar_sfs2 = temp_fs(ar_conts, mzs2)
#redo nd3_sfs based on 206-209 minutes:
#nd3_t = [np.searchsorted(times[0, :], i) for i in [206, 209]]
#nd3_sum = np.mean(fracs[:, nd3_t[0]:nd3_t[1]], axis=1)
#nd3_sum[19] -= nd3_sum[39] * (ar_sfs2[7] / ar_sfs2[10])
#nd3_sum[39] -= nd3_sum[39]
#nd3_conts = get_conts(nd3_sum)
#nd3_sfs2 = temp_fs(nd3_conts, mzs2)
nd3_sfs2 = temp_fs(nd3_sfs, mzs2)
I2_nd3 = I_nd3 * (1 - nd3_sfs[0][1] - nd3_sfs[7][1])
I2_ar = I_ar 
d2_sfs2 = temp_fs(d2_sfs, mzs2)
I2_d2 = I_d2 * (1 - d2_sfs[0][1])

sfs2 = [d2_sfs2, nd3_sfs2, n2_sfs2, ar_sfs2]
Is2 = [I2_d2, I2_nd3, I2_n2, I2_ar]

#normalized arrays
t1, t2 = find_length(0, times[0, :]), find_length(2170, times[0, :])
times2 = times[0, t1:t2]
M2 = np.row_stack([fracs[n - 1, t1:t2] for n in mzs2]) 
A2 = np.row_stack([amus[n - 1, t1:t2] for n in mzs2])
norm_M2 = np.zeros(M2.shape)
M2_tot = np.sum(M2, axis=0)
for i, col in enumerate(M2[0, :]):
    norm_M2[:, i] = M2[:, i] / M2_tot[i]

#for just D, the mzs used should be 2, 3, 4, 6, 14, 16, 18, 20, 22, 28, 36,
#40


fit_M2 = np.zeros(norm_M2.shape).T
params2 = np.zeros((norm_M2.shape[1], 4))
for i, col in enumerate(norm_M2[0, :]):
    if i == 0:
        g, params2[i, :], gM, fit_M2[i, :] = \
                fit_all_justdar(norm_M2[:, i], sfs2, Is2,
                                guesses=[1, 0, 0, 0])
    else:
        g, params2[i, :], gM, fit_M2[i, :] = \
                fit_all_justdar(norm_M2[:, i], sfs2, Is2, guesses=g)
#remove duff values
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
            
#Try to fit the Aux temperature rises
t_fits = [[200, 312], [340, 430], [470, 550], [580, 670], [740, 790],
          [820, 910]]
T_params = []
Tfits = []
for tf in t_fits:
    i_t1, i_t2 = [np.searchsorted(log['Time'].values / 60, t) for t in tf]
    Tnorm = (log['Aux'][i_t1:i_t2] - log['Aux'][i_t1:i_t2].min()) /\
            (log['Aux'][i_t1:i_t2].max() - log['Aux'][i_t1:i_t2].min())
    p, Tfit = expdec_fit_params2(log['Time'].values[i_t1:i_t2] / 60,
                                 1 - Tnorm.values, 
                                 guesses=[1, 0.02, tf[0], -0.05])
    T_params.append(p)
    Tfits.append(Tfit)
    mspec_fit([log['Time'].values[i_t1:i_t2] / 60, Tfit[:, 0]],
              1 - Tnorm.values, Tfit[:, 1], 'Tn', legend=False,
              open_circles=True, colours=['green'], linestyle='dashed',
              opacity=0.05, msize=50)
"""
