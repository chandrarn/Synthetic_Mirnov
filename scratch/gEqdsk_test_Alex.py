from freeqdsk import geqdsk
from sys import path
path.append("/home/rianc/Documents/eqtools-1.0/")
import eqtools as eq

tree = "efit20"
shot = 1120906030
time = 1
eq_f = eq.CModEFIT.CModEFITTree(shot, tree=tree)
eq.filewriter.gfile(
                eq_f, time, nw=200, nh=200, tunit="s", name= f'g{shot}_{time}'
            )

with open(f'g{shot}_{time}', "r") as f:
                eqdsk = geqdsk.read(f)

print(eqdsk)