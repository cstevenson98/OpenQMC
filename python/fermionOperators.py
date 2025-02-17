##
## Copyright (C) 2025 Conor Stevenson
## Licensed under the GNU General Public License v3.0
##
from functools import reduce

from spinOperators import sigmaz_site, identity_n, sigmam_site


def fermi_lower(n, i):
    phase = reduce(lambda a, b: a * b, [sigmaz_site(n, j) for j in range(i)], identity_n(n))
    return phase * sigmam_site(n, i)


def current_i(n, i):
    if i < n:
        cDagc = fermi_lower(n, i).dag() * fermi_lower(n, i+1)
        return cDagc + cDagc.dag()
    else:
        return 0. * identity_n(n)
