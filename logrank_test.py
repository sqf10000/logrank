from numba.pycc import CC

import numpy as np
from numba import njit
from numba import int32, float32,float64
from numba import types, typed
cc = CC('Logrank')
@cc.export('logrank_test', 'float64[:](int64[:], int64[:],int64[:],int64)')

@njit('float64[:](int32[:], int64[:],int64[:],int64)',nogil=True)
def logrank_test(d_myfactor, d_survivalMonth, d_death_observed, d_mylength):
    nall_timetotal = d_mylength
    n2i_timetotal = 0
    d2i_deathtotal = 0  # observed_death
    T2i_sum = 0.0  # expected
    patient_count_death = 0
    V2i_sum = 0.0

    # [HR, chisq, observed, expected]
    for n in range(0, d_mylength, 1):
        if (d_death_observed[n] == 1):
            patient_count_death += 1
        if (d_myfactor[n] == 1):
            n2i_timetotal += 1
            if (d_death_observed[n] == 1):
                d2i_deathtotal += 1
    n = 0
    i = 0
    while i < (d_mylength):
        dall_death_censored = 0
        dall_death = 0
        d2i_death = 0
        d2i_death_censored = 0
        # T2i = 0.0
        # V2i = 0.0
        if ((i == (d_mylength - 1)) or (d_survivalMonth[i] != d_survivalMonth[i + 1])):
            while (n <= i):
                dall_death_censored += 1
                if d_death_observed[n] == 1:
                    dall_death += 1
                if (d_death_observed[n] == 1 and d_myfactor[n] == 1):
                    d2i_death += 1
                if (d_myfactor[n] == 1):
                    d2i_death_censored += 1
                n += 1

            T2i = dall_death / nall_timetotal * n2i_timetotal
            T2i_sum = T2i_sum + T2i
            V2i = (n2i_timetotal / nall_timetotal) * (1 - (n2i_timetotal / nall_timetotal)) * (
                    (nall_timetotal - dall_death) / (nall_timetotal - 0.99999)) * dall_death
            V2i_sum = V2i_sum + V2i

            nall_timetotal = nall_timetotal - dall_death_censored
            n2i_timetotal = n2i_timetotal - d2i_death_censored
        i += 1

    # chisq = (observed_death - expected) * (observed_death - expected) / (myvar + 0.00001)
    chisq = (d2i_deathtotal - T2i_sum) * (d2i_deathtotal - T2i_sum) / (V2i_sum + 0.00001)
    # HR = (observed_death * (patient_count_death - expected)) / (expected * (patient_count_death - observed_death) + 0.00001)
    if V2i_sum == 0:
        HR = 1
    else:
        HR = (d2i_deathtotal * (patient_count_death - T2i_sum)) / (
                T2i_sum * (patient_count_death - d2i_deathtotal) + 0.00001)
    # [HR, chisq, observed, expected]
    d_hrresult = [HR, chisq, d2i_deathtotal, T2i_sum]
    #pvalue = chisq
    # pvalue = chi2.sf(chisq, 1)
    # pvalue chi2.sf(d_hrresult[1], 1)
    re = np.array([1.0, 1.0, 0.0, 0.0])

    re[0] = HR
    re[1] = chisq
    re[2] = np.count_nonzero(d_myfactor == 0)
    re[3] = (d_myfactor == 1).sum()
    return re
if __name__ == "__main__":
    cc.compile()
