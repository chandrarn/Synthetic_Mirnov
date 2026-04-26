# generate quick q(r) profile
from sys import path
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
import numpy as np

path.append('/home/rianc/Documents/eqtools-1.0/')
import eqtools as eq
from freeqdsk import geqdsk
shotno = 1120906030

eq_f = eq.CModEFIT.CModEFITTree(shotno, tree = 'efit21')

print(eq_f.getQProfile().shape)

q = eq_f.getQProfile()
psi = eq_f.getRmidPsi()
time = eq_f.getTimeBase()

plt.figure(figsize=(4,3),layout='constrained', num=f'q profile for shot {shotno}')
target_times = [1,1.2,1.4]

psi_to_psiN = lambda psi: np.sqrt( (psi - psi[0])/(psi[-1] - psi[0]) )

for t in target_times:
    idx = (np.abs(time-t)).argmin()
    plt.plot(psi_to_psiN(psi[idx]),q[idx],label=f't={time[idx]:.2f}s')
plt.xlabel('$\psi_N$')
plt.ylabel('$q$')
plt.title(f'{shotno} q Profile')
plt.grid()
plt.legend()


plt.savefig(f'q_profile_{shotno}.pdf')
print('Saved '+f'q_profile_{shotno}.pdf')


print("done")

###################################
# # import sys
# # sys.path.append('/home/rianc/Documents/eqtools-1.0') 
# from eqtools.CModEFIT import CModEFITTree
# from pathlib import Path

# shot = 1140815006
# tree = "efit20" # 'efit21' doesn't work for some reason
# time = 0.500

# onsims_path = Path(__file__).resolve().parent.parent
# eqdsk_path = f"{onsims_path}/data_files/inputs/shot_{shot}/eqdsk/geqdsk_{int(time*1e3):05d}.eqdsk"

# e = CModEFITTree(shot, tree=tree)
# e.gfile(time=time, name=eqdsk_path)