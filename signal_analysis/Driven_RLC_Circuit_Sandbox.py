#Sandbox for RLC circuit analysis in python
import numpy as np
import matplotlib.pyplot as plt

# RLC circuit parameters
R = 30      # Ohms
L = 1e-4      # Henry
C = 1e-9     # Farads

# Frequency range
f = np.logspace(1, 6, 1000)  # Hz
w = 2 * np.pi * f                 # Angular frequency

# Transfer function H(w) for series RLC (output across the capacitor)
Z_R = R
Z_L =  (1j * w * L)
Z_C = 1 / (1j * w * C)
Z_total = Z_R + Z_L + Z_C

H = Z_C / Z_total  # Voltage across capacitor / input voltage

# Magnitude and phase response
mag = np.abs(H)
phase = np.angle(H, deg=True)

# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(f, mag)
ax[0].set_ylabel('Magnitude')
ax[0].set_title('RLC Circuit Frequency Response')
ax[0].grid(True)

ax[1].plot(f, phase)
ax[1].set_ylabel('Phase (degrees)')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].grid(True)

plt.tight_layout()
plt.show()