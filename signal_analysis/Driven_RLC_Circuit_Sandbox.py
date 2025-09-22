#Sandbox for RLC circuit analysis in python
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# RLC circuit parameters
R = 7      # Ohms
L = 60e-6      # Henry
C = 700e-12     # Farads

# Frequency range
f = np.logspace(1, 6, 1000)  # Hz
w = 2 * np.pi * f                 # Angular frequency
 
# Transfer function H(w) for series RLC (output across the capacitor)
Z_R = lambda w: R 
def Z_R_dynam(w):
    out = R * np.ones_like(w)
    out[w/(2*np.pi)>=220e3] *= 1+(np.sqrt( w[w/(2*np.pi)>=220e3]/(2*np.pi))-np.sqrt(220e3))*0.0043 # Skin depth correction
    return out
Z_L =  (1j * w * L)
Z_C = 1 / (1j * w * C)
Z_total_dynam = Z_R_dynam(w) + Z_L + Z_C
Z_total = Z_R(w) + Z_L + Z_C

H_dynam = Z_C / Z_total_dynam  # Voltage across capacitor / input voltage
H = Z_C / Z_total  # Voltage across capacitor / input voltage

# Magnitude and phase response
mag_dynam = np.abs(H_dynam)
phase_dynam = np.angle(H_dynam, deg=True)
mag = np.abs(H)
phase = np.angle(H, deg=True)

# Plot
fig, ax = plt.subplots(2, 1, figsize=(4, 8), sharex=True)
ax[0].plot(f*1e-6, mag,label=r'R=%d $\Omega$, L=%.1f $\mu$H, C=%.1f pF'%(R,L*1e6,C*1e12))
ax[0].plot(f*1e-6, mag_dynam,label='Dynamic skin depth R')
ax[0].set_ylabel('Magnitude')
ax[0].set_title('RLC Circuit Frequency Response')
ax[0].grid(True)
ax[0].legend(fontsize=8,loc='upper right')

ax[1].plot(f*1e-6, phase)
ax[1].plot(f*1e-6, phase_dynam)
ax[1].set_ylabel('Phase (degrees)')
ax[1].set_xlabel('Frequency (MHz)')
ax[1].grid(True)

fig.savefig('../output_plots/RLC_Circuit_Response.pdf',dpi=300,transparent=True)

plt.tight_layout()
plt.show()

print('Done')