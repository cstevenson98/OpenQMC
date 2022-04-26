import numpy as np
from matplotlib import pyplot as plt
from qutip import lindblad_dissipator, fock, tensor, mesolve, steadystate, to_super, expect, operator_to_vector, \
    sprepost

from fermiChain import fermi_tight_binding, fermi_lower
from fermionOperators import current_i
from spinOperators import identity_n


def comm(a, b):
    return a*b - b*a

def anticomm(a, b):
    return a*b + b*a

def supercomm(H, n):
    return sprepost(H, identity_n(n)) - sprepost(identity_n(n), H)

if __name__ == '__main__':
    n = 4

    H = fermi_tight_binding(n, 20., 1., 1.)
    c = [fermi_lower(n, i) for i in range(n)]

    times = np.linspace(0.0, 200.0, 2000)
    psi0 = tensor([fock(2, 0) for i in range(n)]) + tensor([fock(2, 1) for i in range(n)])
    psi0 /= psi0.norm()

    collapse_ops = [.5 * c[0].dag(), .75 * c[3]]

    rho_ss = steadystate(H, collapse_ops, method='power', use_rcm=True)

    print('total current', expect(current_i(n, 0)+current_i(n, 1)+current_i(n, 2), rho_ss))

    result = mesolve(H, psi0, times, collapse_ops, [current_i(n, 0) + current_i(n, 1) + current_i(n, 2)])

    plt.figure()
    plt.plot(times, result.expect[0])

    plt.show()
