from functools import reduce

from fermionOperators import fermi_lower
from spinOperators import zeros_n


def fermi_tight_binding(n, mu0, J, delta):
    # H0 =  µ0 ∑_i  ci† ci
    H_0 = mu0 * reduce(lambda a, b: a + b, [fermi_lower(n, i).dag() * fermi_lower(n, i) for i in range(n)], zeros_n(n))

    # Hhop =  -J ∑_i  ci† ci+1 + h.c.
    H_hop = -J * reduce(lambda a, b: a + b, [fermi_lower(n, i).dag() * fermi_lower(n, i + 1) for i in range(n - 1)],
                   zeros_n(n))
    H_hop += H_hop.dag()

    # Hhop =  -∆ ∑_i  ci† ci+1 + h.c.
    H_aniso = -delta * reduce(lambda a, b: a + b, [fermi_lower(n, i) * fermi_lower(n, i + 1) for i in range(n - 1)], zeros_n(n))
    H_aniso += H_aniso.dag()

    return H_0 + H_hop + H_aniso
