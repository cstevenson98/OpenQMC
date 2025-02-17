##
## Copyright (C) 2025 Conor Stevenson
## Licensed under the GNU General Public License v3.0
##
from qutip import qeye, sigmax, tensor, sigmay, sigmaz, sigmap, sigmam, qzero


def zeros_n(n):
    twos = [2 for i in range(n)]
    return qzero(twos)


def identity_n(n):
    twos = [2 for i in range(n)]
    return qeye(twos)


def operator_site(op, i, j):
    return qeye(2) if i != j else op


def sigmax_site(n, i):
    return tensor([operator_site(sigmax(), i, j) for j in range(n)])


def sigmay_site(n, i):
    return tensor([operator_site(sigmay(), i, j) for j in range(n)])


def sigmaz_site(n, i):
    return tensor([operator_site(sigmaz(), i, j) for j in range(n)])


def sigmap_site(n, i):
    return tensor([operator_site(sigmap(), i, j) for j in range(n)])


def sigmam_site(n, i):
    return tensor([operator_site(sigmam(), i, j) for j in range(n)])
