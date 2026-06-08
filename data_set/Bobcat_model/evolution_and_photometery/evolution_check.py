import numpy as np
from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt

evo00_path = '/Users/onnalininnala/Intern/synthesis/Bobcat/evolution_and_photometery/evolution_tables/evo_tables+0.0/nc+0.0_co1.0_mass_age'
evo05_path = '/Users/onnalininnala/Intern/synthesis/Bobcat/evolution_and_photometery/evolution_tables/evo_tables+0.5/nc+0.5_co1.0_mass_age'

header = 'Teff logg Mass Radius logL logAge\n'

# Read and fix header mh=0.0
with open(evo00_path) as f:
    lines00 = f.readlines()[2:]
evo00_data = ascii.read([header] + lines00, delimiter=' ', guess=False)
points00 = np.column_stack((evo00_data['Teff'], evo00_data['logg']))
radius00 = evo00_data['Radius']

# Read and fix header mh=0.5
with open(evo05_path) as f:
    lines05 = f.readlines()[2:]
evo05_data = ascii.read([header] + lines05, delimiter=' ', guess=False)
points05 = np.column_stack((evo05_data['Teff'], evo05_data['logg']))
radius05 = evo05_data['Radius']

# Compare and compute delta radius
if np.array_equal(points00, points05):
    delta_radius = (radius05 - radius00) * 100 / radius00 
print(delta_radius)

# --- Data ---
teff = evo00_data['Teff']
logg = evo00_data['logg']
x = radius00
y = radius05
z = delta_radius  # percentage difference

# --- Figure with 2 subplots (1 row, 2 columns) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

# --- Left: Teff vs logg contour ---
cntr = ax1.tricontourf(teff, logg, z, levels=20, cmap="viridis")
ax1.scatter(teff, logg, c='k', s=5, alpha=0.5)  # actual points
ax1.set_xlabel("Teff [K]")
ax1.set_ylabel("log g [cgs]")
ax1.set_title("Δ Radius Contours (mh=+0.5 vs +0.0)")
ax1.invert_yaxis()

# Add colorbar for left plot
cbar1 = fig.colorbar(cntr, ax=ax1)
cbar1.set_label(r'$\Delta$ Radius [%]')

# --- Right: Radius–Radius scatter ---
sc = ax2.scatter(x, y, c=z, cmap="viridis", s=30, edgecolor="k")
ax2.plot([min(x), max(x)], [min(x), max(x)], 'r--', label="1:1 line")
ax2.set_xlabel(r"Radius ($R_\odot$) (mh = +0.0)")
ax2.set_ylabel(r"Radius ($R_\odot$) (mh = +0.5)")
ax2.set_title("Radius Comparison (Color = Δ Radius %)")
ax2.legend()

# Add colorbar for right plot
cbar2 = fig.colorbar(sc, ax=ax2)
cbar2.set_label(r'$\Delta$ Radius [%]')

plt.tight_layout()
plt.savefig('/Users/onnalininnala/Intern/synthesis/Bobcat/evolution_and_photometery/evolution_tables/delta_evo.png', dpi=300)
