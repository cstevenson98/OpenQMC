import numpy as np
from matplotlib import pyplot as plt
from qutip import lindblad_dissipator, fock, tensor, mesolve, steadystate, to_super, expect, operator_to_vector, \
    sprepost

from fermiChain import fermi_tight_binding, fermi_lower
from spinOperators import identity_n


def comm(a, b):
    return a*b - b*a

def anticomm(a, b):
    return a*b + b*a

def supercomm(H, n):
    return sprepost(H, identity_n(n)) - sprepost(identity_n(n), H)

if __name__ == '__main__':
    n = 4

    H = fermi_tight_binding(n, 1., 2., 5.)
    c = [fermi_lower(n, i) for i in range(n)]

    times = np.linspace(0.0, 20.0, 2000)
    psi0 = tensor([fock(2, 0) for i in range(n)]) + tensor([fock(2, 1) for i in range(n)])
    psi0 /= psi0.norm()

    collapse_ops = [c[0].dag(), c[3]]

    # Pumping site-1, sink site-N
    L1 = lindblad_dissipator(c[0].dag(), c[0].dag())
    L4 = lindblad_dissipator(c[3], c[3])

    rho_ss = steadystate(H, collapse_ops, method='power', use_rcm=True)

    print(c[0].dag() * c[0])

    result = mesolve(H, psi0, times, [L1, L4], [c[0].dag()*c[0], c[1].dag()*c[1], c[2].dag()*c[2], c[3].dag()*c[3]])

    plt.figure()
    plt.plot(times, result.expect[0])
    plt.plot(times, result.expect[1])
    plt.plot(times, result.expect[2])
    plt.plot(times, result.expect[3])

    plt.show()
