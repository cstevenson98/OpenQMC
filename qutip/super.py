from qutip import *

Sm1 = tensor(sigmam(), identity(2))
Sm2 = tensor(identity(2), sigmam())

print(Sm2)
# print(2 * lindblad_dissipator(Sm2))

A = 2*tensor(Sm2, Sm2)
B = tensor(Sm2.dag()*Sm2, identity([2, 2]))
C = tensor(identity([2, 2]), Sm2.dag()*Sm2)

print(B+C)
