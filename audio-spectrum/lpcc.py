import numpy as np
import scipy.linalg
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import glob


def autocorr(seq, order=None):
    if order is None:
        order = len(seq) - 1
    autocor = []
    for tau in range(order + 1):
        s = np.sum([seq[n] * seq[n + tau] for n in range(len(seq) - tau)])
        autocor.append(s)
    return autocor


def lpc(seq, order=None):
    acseq = np.array(autocorr(seq, order))
    a_coef = np.dot(np.linalg.pinv(scipy.linalg.toeplitz(acseq[:-1])), -acseq[1:].T)
    err_term = acseq[0] + np.dot(acseq[1:], a_coef)
    return a_coef.tolist(), np.sqrt(abs(err_term))


def lpcc(seq, err_term, order=None):
    if order is None:
        order = len(seq) - 1
    lpcc_coeffs = [np.log(err_term), -seq[0]]
    for n in range(2, order + 1):
        upbound = order + 1 if n > order else n
        lpcc_coef = -sum(i * lpcc_coeffs[i] * seq[n - i - 1] for i in range(1, upbound)) / upbound
        if n <= len(seq):
            lpcc_coef -= seq[n - 1]
        lpcc_coeffs.append(lpcc_coef)
    return lpcc_coeffs


order = 12
for count, filename in enumerate(glob.glob('1272-128104-0000.flac')):
    sr, wav = read(filename)
    lpc_value, err = lpc(wav, order)
    lpcc_value = lpcc(lpc_value, err, order)

    print(lpcc_value)

    plt.figure(count)
    plt.xlabel("Time")
    plt.ylabel("LPCC Coeff")
    plt.plot(lpcc_value)
    plt.show()
