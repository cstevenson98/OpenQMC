from functools import reduce

from spinOperators import sigmaz_site, identity_n, sigmam_site


def fermi_lower(n, i):
    phase = reduce(lambda a, b: a * b, [sigmaz_site(n, j) for j in range(i)], identity_n(n))
    return phase * sigmam_site(n, i)
