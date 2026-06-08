import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.visualization import make_lupton_rgb
import os
import numpy as np
from astropy.table import Table, vstack, Column, MaskedColumn
import pandas as pd
import math
import statistics
from scipy.interpolate import interp1d, RBFInterpolator
import warnings
from collections import Counter
import random
from astropy.cosmology import FlatLambdaCDM
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.stats import norm
import re
from datetime import datetime
import json
import xarray

def cm_to_pc(cm):
    return cm / 3.085677581e18

def pc_to_km(pc):
    return pc * 3.085677581e13

def get_spectral_type_from_temp(temp_input):
        # ---temperature for each type (Fig22 in Kirkpatrick2021)
        type_temp = {
            'M7': 2708, 'M8': 2535, 'M9': 2362, 'L0': 2212, 'L1': 2087,
            'L2': 1969, 'L3': 1843, 'L4': 1709, 'L5': 1615, 'L6': 1512,
            'L7': 1434, 'L8': 1324, 'L9': 1269, 'T0': 1261, 'T1': 1238,
            'T2': 1223, 'T3': 1199, 'T4': 1184, 'T5': 1121, 'T6': 971,
            'T7': 838, 'T8': 688, 'T9': 578, 'Y0': 484, 'Y1': 382, 'Y2': 287
        }
        sorted_items = sorted(type_temp.items(), key=lambda x: -x[1])
        types, temps = zip(*sorted_items)
        boundaries = [(temps[i] + temps[i + 1]) / 2 for i in range(len(temps) - 1)]
        def classify(temp):
            for i, boundary in enumerate(boundaries):
                if temp >= boundary:
                    return types[i]
            return types[-1]
        if isinstance(temp_input, (list, np.ndarray)):
            return [classify(temp) for temp in temp_input]
        else:
            return classify(temp_input)

def create_folders(parameters, output_path, m_detection_arr, imove):
    m_detection = m_detection_arr[imove] #mag that want to move
    #creating output folders
    moveflux_folder = f"{output_path}/{parameters['fields_name']}"
    output_folder = moveflux_folder+f'/mag_{m_detection}'
    if not os.path.exists(output_folder):
        os.makedirs(moveflux_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
    return moveflux_folder, output_folder

def extracts_flux_bobcat(parameters):
    Rsun_cm = 6.9599E+10 #cm
    def is_text_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read the first 1KB
            return True
        except UnicodeDecodeError:
            return False
    
    def read_planet_params_bobcat_from_txt_line(line):
            values = line.strip().split()[:8]
            teff, grav_mks, Y, f_rain, Kzz = map(float, values[:5])
            feh_str = values[5]          # Keep original string for feh
            co_ratio, f_hole = map(float, values[6:8])

            # Convert grav [m/s²] → logg in cgs units
            logg = np.log10(grav_mks * 100)  # m/s² → cm/s²
            logkzz = np.log10(Kzz)

            # Format feh to always have sign:
            feh_float = float(feh_str)
            if feh_float >= 0:
                feh_formatted = f"+{feh_float}"
            else:
                feh_formatted = f"{feh_float}"

            return logg, teff, feh_formatted, co_ratio, logkzz, f_hole

    def read_flux_table_bobcat(lines):
        wavelengths = []
        fluxes = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.lower().startswith('micron'):
                continue  # Skip header, empty lines, or comment lines
            try:
                parts = line.split()
                wavelengths.append(float(parts[0]))
                fluxes.append(float(parts[1]))
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue
        return np.array(wavelengths), np.array(fluxes)
    
    
    ################## Bobcat ##############################
    program_path = parameters.get('enzo_path') or os.getcwd()
    main_fol = os.path.join(program_path, "Results", parameters.get('stellar_model'))
    data_fol = os.path.join(main_fol,  parameters["fields_name"])

    if not os.path.exists(data_fol):
        os.makedirs(main_fol, exist_ok=True)
        os.makedirs(data_fol)

    all_fil = parameters['bands']
    
    print(f'start to create photometry catalog from synthesis SEDs of Sonora Bobcat evolution')
    mh_list = parameters['mh_list']
    for mh_dir in mh_list:
        folder_path = f"{main_fol}/data_set/Bobcat_model/spectra/spectra_m{mh_dir}"
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        planet_files = [f for f in os.listdir(folder_path)]
        names, loggs, teffs, mhs, ctos, logkzzs, ph3s, radius, types = [], [], [], [], [], [], [], [], []
        fluxes_per_filter = {filt: [] for filt in parameters['bands']}
        mags_per_filter = {filt: [] for filt in parameters['bands']}
        for file_name in planet_files:
            full_path = os.path.join(folder_path, file_name)
            name = file_name.replace("sp_", "", 1)
            # Skip non-text (binary) files
            if not is_text_file(full_path):
                print(f"Skipping non-text file: {file_name}")
                continue
            with open(full_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
                if not lines:
                    continue
                # Read first line = planet params
                logg, teff_val, mh, cto, logkzz, ph3 = read_planet_params_bobcat_from_txt_line(lines[0])
                lam_spec, flux_spec = read_flux_table_bobcat(lines[1:])  # skip param line

                # Radius estimation
                m_val = float(name.split('_m')[1])
                m_str = f"{m_val:+.1f}"
                evo_path = f"{main_fol}/data_set/Bobcat_model/evolution_and_photometery/evolution_tables/evo_tables{m_str}/nc{m_str}_co1.0_mass_age"
                with open(evo_path) as f:
                    lines = f.readlines()[2:]
                header = 'Teff logg Mass Radius logL logAge\n'
                evo_data = ascii.read([header] + lines, delimiter=' ', guess=False)
                points = np.column_stack((evo_data['Teff'], evo_data['logg']))
                values_radius = evo_data['Radius']

                radius_func = RBFInterpolator(points, values_radius, kernel='thin_plate_spline')  # 'linear', 'cubic', 'quintic', 'thin_plate_spline'
                radius_est = radius_func([[teff_val, logg]])[0]
                radius_cm = radius_est * Rsun_cm
                radius_pc = cm_to_pc(radius_cm)

                spectral_type = get_spectral_type_from_temp(teff_val)

                names.append(name)
                loggs.append(logg)
                teffs.append(teff_val)
                mhs.append(mh)
                ctos.append(cto)
                logkzzs.append(logkzz)
                ph3s.append(ph3)
                radius.append(radius_pc)
                types.append(spectral_type)

                # Read flux table
                lam_spec_cm= lam_spec * 1e-4  # micron to cm
                flux_spec_cgs = flux_spec #f_nu_cgs (erg/cm^2/s/Hz)
                for filt in all_fil:
                    lam_filt_aas, response_filt = np.loadtxt(
                        f"{main_fol}/data_set/transmission_filters/{parameters['filters_program']}_{filt}.dat",
                        unpack=True)
                    lam_filt_cm = lam_filt_aas * 1e-8 # angstrom to cm
                    response_int = np.interp(lam_spec_cm, lam_filt_cm, response_filt, left=0, right=0)
                    numerator = np.trapezoid(response_int * lam_spec_cm * flux_spec_cgs, lam_spec_cm)
                    denominator = np.trapezoid(response_int * lam_spec_cm, lam_spec_cm)
                    F_nu = numerator / denominator  # f_nu_cgs erg/cm^2/s/Hz

                    # AB magnitude
                    m_ab = -2.5 * np.log10(F_nu) - 48.6
                    fluxes_per_filter[filt].append(F_nu)
                    mags_per_filter[filt].append(m_ab)

                mag_col = [names, loggs, teffs, mhs, ctos, logkzzs, ph3s, radius, types]
                col_names = ["name", "logg", "Teff", "mh", "cto", "logkzz", "PH3", "Radius_pc", "SpT"]
                for filt in all_fil:
                    mag_col.append(mags_per_filter[filt])
                    col_names.append(f"f_{filt}")

        mag_table = Table(mag_col, names=col_names)
        output_m = os.path.join(data_fol, f"mag_{parameters['filters_program']}_m{mh_dir}.dat")
        ascii.write(mag_table, output_m, overwrite=True, delimiter='\t')
            
        print('___________________')
        print(f"Saved: {mh_dir}")
        print('___________________')

def extracts_flux_elfowl(parameters):
    """
    Reads model spectra, computes fluxes and magnitudes in specified filters,
    interpolates radii from evolutionary tables, and saves flux & magnitude tables.

    parameters: dict with keys:
        - output_path: str, path for output
        - bands: list of filter names (e.g., ['F555W','F606W',...])
        - filters_profile_path: str, path to filter transmission curves
    """
    import os
    import json
    import numpy as np
    import xarray
    from astropy.table import Table
    from astropy.io import ascii
    from scipy.interpolate import RBFInterpolator
    from astropy.table import MaskedColumn

    # --- Constants ---
    Rsun_cm = 6.9599E+10
    c = 2.99792458e10
    pc_in_cm = 3.085677581e18

    # --- Helper functions ---
    def cm_to_pc(cm):
        return cm / pc_in_cm

    def read_planet_params_elf_olw(data):
        planet_params = json.loads(data.attrs['planet_params'])
        mh_val = planet_params["mh"]
        if isinstance(mh_val, str):
            mh_val = float(mh_val)

        # Convert grav [m/s²] → logg in cgs units
        grav_mks = planet_params["logg"]["value"]
        logg_cms2 = np.log10(grav_mks * 100) # m/s² → cm/s²
        return (
            logg_cms2,
            planet_params["teff"]["value"],
            mh_val,
            planet_params["cto"],
            planet_params["logkzz"],
            planet_params["PH3"]
        )

    # --- Setup paths ---
    program_path = parameters.get('enzo_path') or os.getcwd()
    main_fol = os.path.join(program_path, "Results", parameters.get('stellar_model'))
    data_fol = f"{main_fol}/{parameters['fields_name']}"
    os.makedirs(data_fol, exist_ok=True)

    all_fil = parameters['bands']
    program = parameters['filters_program']

    if program == 'HST':
        lamda_effs = {'F300X':2879.28, 'F475X':4855.99,
            'F555W': 5331.75, 'F606W': 5809.26, 'F625W': 6266.20, 'F775W': 7652.44, 
            'F814W': 7973.39, 'F850LP': 9004.99, 'F098M': 9826.81, 'F105W': 10430.83, 
            'F110W': 11200.52, 'F125W': 12363.55, 'F140W': 13734.66, 'F160W': 15278.47,
            'F350LP': 5546.11, 'F435W': 4341.62, 'F475W': 4708.87, 'F600LP': 7291.35}
        
    elif program == 'Roman':
        lamda_effs = {'F062': 6141.54, 'F087':8650.97, 'F106':10465.04, 'F129':12759.99, 'F158':15577.83,
            'F184':18290.96, 'F146':13049.63, 'F213':21116.87,}
        
    elif program == 'JWST':
        lamda_effs = {'F070W':6988.43, 'F090W':8984.98, 'F115W':11433.62, 'F150W':14872.56 , 
                'F200W':19680.41, 'F277W':27278.58, 'F300M': 29818.32,'F356W':35287.04,
                'F410M':40723.18, 'F444W':43504.26, 'F480M': 48139.11}

    spectra_path = ""#path of ELF OWL SEDs
    teff_list = [f for f in os.listdir(spectra_path) if not f.startswith('._')]

    mh_list = parameters['mh_list']

    for mh_val in mh_list:
        # Ensure mh_val is numeric
        mh_float = float(mh_val)

        # Format with sign for folder/file names
        mh_str = f"{mh_float:+.1f}"  # e.g., +0.0, -0.5, +0.5

        # Storage
        names, loggs, teffs, mhs, ctos, logkzzs, ph3s = [], [], [], [], [], [], []
        fluxes_per_filter = {filt: [] for filt in all_fil}
        mags_per_filter = {filt: [] for filt in all_fil}

        # Loop over Teff folders
        for teff in teff_list:
            folder_path = f"{spectra_path}/{teff}"
            if not os.path.exists(folder_path):
                continue

            # Keep only files with the correct mh value in the name
            planet_files = [
                f for f in os.listdir(folder_path)
                if f.endswith('.nc')
                and not f.startswith('._')
                and f"mh_{mh_float}" in f
            ]

            for file_name in planet_files:
                ds = xarray.load_dataset(f"{folder_path}/{file_name}", engine="netcdf4")
                logg, teff_val, mh, cto, logkzz, ph3 = read_planet_params_elf_olw(ds)

                names.append(file_name)
                loggs.append(logg)
                teffs.append(teff_val)
                mhs.append(mh)
                ctos.append(cto)
                logkzzs.append(logkzz)
                ph3s.append(ph3)

                lam_spec = ds["wavelength"].values * 1e-4  # micron → cm
                flux_spec = ds["flux"].values #erg/cm**2/s/cm

                for filt in all_fil:
                    lam_filt_aas, response_filt = np.loadtxt(
                        f"{main_fol}/data_set/transmission_filters/{parameters['filters_program']}_{filt}.dat",
                        unpack=True)
                    lam_filt = lam_filt_aas * 1e-8  # Å → cm
                    filt_int = np.interp(lam_spec, lam_filt, response_filt, left=0, right=0)
                    numerator = np.trapezoid(filt_int * lam_spec * flux_spec, lam_spec)
                    denominator = np.trapezoid(filt_int * lam_spec, lam_spec)
                    flux_lam = numerator / denominator  # erg/s/cm²/cm
                    lamda_eff_filt = lamda_effs[filt] * 1.e-8
                    F_nu = (lamda_eff_filt ** 2) * flux_lam / c
                    m_ab = -2.5 * np.log10(F_nu) - 48.6

                    fluxes_per_filter[filt].append(F_nu)
                    mags_per_filter[filt].append(m_ab)

        # --- Build tables ---
        col_names = ["name", "logg", "Teff", "mh", "cto", "logkzz", "PH3"] + [f"f_{f}" for f in all_fil]
        flux_data = [names, loggs, teffs, mhs, ctos, logkzzs, ph3s] + [fluxes_per_filter[f] for f in all_fil]
        mag_data = [names, loggs, teffs, mhs, ctos, logkzzs, ph3s] + [mags_per_filter[f] for f in all_fil]

        flux_table = Table(flux_data, names=col_names)
        mag_table = Table(mag_data, names=col_names)

        # Convert flux from ZP=-48.6 to ZP=25
        for filt in all_fil:
            col = f"f_{filt}"
            flux_table[col] = 10 ** (((( -2.5 * np.log10(flux_table[col]) ) - 48.6) - 25) / -2.5)

        flux_table["Radius_pc"] = np.zeros(len(flux_table))
        flux_table["SpT"] = MaskedColumn([''] * len(flux_table), dtype='U10')

        mag_table["Radius_pc"] = np.zeros(len(mag_table))
        mag_table["SpT"] = MaskedColumn([''] * len(flux_table), dtype='U10')

        # Filter rows for this mh
        flux_sub = flux_table[flux_table["mh"] == mh_float].copy()
        mag_sub = mag_table[mag_table["mh"] == mh_float].copy()

        if mh_str in ["-0.5", "+0.0", "+0.5"]:
            evo_path = f"{main_fol}/data_set/Bobcat_model/evolution_and_photometery/evolution_tables/evo_tables{mh_str}/nc{mh_str}_co1.0_mass_age"
            with open(evo_path) as f:
                lines = f.readlines()[2:]
            header = "Teff logg Mass Radius logL logAge\n"
            evo_data = ascii.read([header] + lines, delimiter=' ', guess=False)
            points = np.column_stack((evo_data['Teff'], evo_data['logg']))
            values_radius = evo_data['Radius']

            radius_func = RBFInterpolator(points, values_radius, kernel='thin_plate_spline')

            for idx in range(len(flux_sub)):
                teff_candi = flux_sub["Teff"][idx]
                logg_candi = flux_sub["logg"][idx]
                radius_est = radius_func([[teff_candi, logg_candi]])[0]
                radius_cm = radius_est * Rsun_cm
                radius_pc = cm_to_pc(radius_cm)

                sp_type = get_spectral_type_from_temp(teff_candi)
                #sp_group = map_sp_type_group(sp_type)

                flux_sub["Radius_pc"][idx] = radius_pc
                flux_sub["SpT"][idx] = sp_type
                mag_sub["Radius_pc"][idx] = radius_pc
                mag_sub["SpT"][idx] = sp_type
        else:
            mh = float(mh_str)   # convert back to number
            mh_sign = "+" if mh >= 0 else "-"
            evo_path = f"{main_fol}/data_set/Bobcat_model/evolution_and_photometery/evolution_tables/evo_tables{mh_sign}0.5/nc{mh_sign}0.5_co1.0_mass_age"
            with open(evo_path) as f:
                lines = f.readlines()[2:]
            header = "Teff logg Mass Radius logL logAge\n"
            evo_data = ascii.read([header] + lines, delimiter=' ', guess=False)
            points = np.column_stack((evo_data['Teff'], evo_data['logg']))
            values_radius = evo_data['Radius']

            radius_func = RBFInterpolator(points, values_radius, kernel='thin_plate_spline')

            for idx in range(len(flux_sub)):
                teff_candi = flux_sub["Teff"][idx]
                logg_candi = flux_sub["logg"][idx]
                radius_est = radius_func([[teff_candi, logg_candi]])[0]
                radius_cm = radius_est * Rsun_cm
                radius_pc = cm_to_pc(radius_cm)

                sp_type = get_spectral_type_from_temp(teff_candi)

                flux_sub["Radius_pc"][idx] = radius_pc
                flux_sub["SpT"][idx] = sp_type
                mag_sub["Radius_pc"][idx] = radius_pc
                mag_sub["SpT"][idx] = sp_type

        new_order = ["name", "logg", "Teff", "mh", "cto", "logkzz", "PH3", "Radius_pc", "SpT"] + [f"f_{f}" for f in all_fil]
        flux_sub = flux_sub[new_order]
        mag_sub = mag_sub[new_order]

        ascii.write(flux_sub, os.path.join(data_fol, f"flux_zp25_{parameters['filters_program']}_m{mh_str}.dat"), overwrite=True, delimiter="\t")
        ascii.write(mag_sub, os.path.join(data_fol, f"mag_{parameters['filters_program']}_m{mh_str}.dat"), overwrite=True, delimiter="\t")
        print(f"Saved: mh_{mh_str}")

def sort_flux(parameters):
    """
    Read Elf Owl photometric catalogs and save cleaned
    magnitude and flux tables.

    Output:
        mag_<program>_m+0.0.dat
        flux_<program>_m+0.0.dat
    """

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    main_fol = parameters["enzo_path"]

    data_fol = os.path.join(
        main_fol,
        "Results",
        parameters["stellar_model"],
        parameters["fields_name"],
    )

    os.makedirs(data_fol, exist_ok=True)

    all_fil = parameters["bands"]
    mh_list = parameters["mh_list"]

    # -------------------------------------------------
    # Fixed columns
    # -------------------------------------------------
    fixed_cols = ["name", "logg", "Teff", "mh",
        "cto", "logkzz", "PH3", "Radius_pc", "SpT",]

    # -------------------------------------------------
    # Loop over metallicities
    # -------------------------------------------------
    for mh_val in mh_list:

        mh_float = float(mh_val)
        mh_str = f"{mh_float:+.1f}"

        # =====================================================
        # MAGNITUDE TABLE
        # =====================================================

        mag_input = os.path.join(
            main_fol,
            "data_set",
            "Elf_Owl_photometry",
            f"mag_{parameters['filters_program']}_m{mh_str}.dat",  #_cto1.0
        )

        if os.path.exists(mag_input):

            mag_tab = ascii.read(mag_input)

            mag_cols = (
                fixed_cols
                + [f"f_{f}" for f in all_fil]
            )

            available_cols = [
                c for c in mag_cols
                if c in mag_tab.colnames
            ]

            mag_out = mag_tab[available_cols]

            mag_output = os.path.join(
                data_fol,
                f"mag_{parameters['filters_program']}_m{mh_str}.dat",
            )

            ascii.write(
                mag_out,
                mag_output,
                overwrite=True,
                delimiter="\t",
            )

            print(f"Saved: {mag_output}")

        else:
            print(f"WARNING: Missing {mag_input}")

        # =====================================================
        # FLUX TABLE
        # =====================================================

        flux_input = os.path.join(
            main_fol,
            "data_set",
            "Elf_Owl_photometry",
            f"flux_zp25_{parameters['filters_program']}_m{mh_str}.dat", #_cto1.0
        )

        if os.path.exists(flux_input):

            flux_tab = ascii.read(flux_input)

            flux_cols = fixed_cols

            for filt in all_fil:
                if f"f_{filt}" in flux_tab.colnames:
                    flux_cols.append(f"f_{filt}")

                if f"e_{filt}" in flux_tab.colnames:
                    flux_cols.append(f"e_{filt}")

            flux_out = flux_tab[flux_cols]

            flux_output = os.path.join(
                data_fol,
                f"flux_zp25_{parameters['filters_program']}_m{mh_str}.dat",
            )

            ascii.write(
                flux_out,
                flux_output,
                overwrite=True,
                delimiter="\t",
            )

            print(f"Saved: {flux_output}")

        else:
            print(f"WARNING: Missing {flux_input}")

def generate_mock_catalog(parameters, m_detec_set, output_path, mh):
    """Generate a mock catalog with adjusted magnitudes for each band."""
    c_AAs = 2.99792458e18  # Speed of light in Angstrom/s
    ##############################################################
    print(f'Starting mock catalog generation at detection magnitude: {m_detec_set}')
    output_m = os.path.join(output_path, f"{parameters['fields_name']}/mag_{parameters['filters_program']}_m{mh}.dat")
    mag_ori = ascii.read(output_m)
    curmocktable = mag_ori.copy()  # to be kept

            ## RA: 0 to 360 degrees
    parameters['solid_angle'][0] = float(parameters['solid_angle'][0])
    image_width = np.sqrt(parameters['solid_angle'][0])
    fake_ra = np.random.uniform((parameters['RA'] - (image_width/ 2)), (parameters['RA'] + (image_width / 2)), size=len(curmocktable))

            # DEC: -90 to +90 degrees
    fake_dec = np.random.uniform((parameters['DEC'] - (image_width / 2)), (parameters['DEC'] + (image_width / 2)), size=len(curmocktable))

    # Add to the table
    curmocktable['RA_fake'] = fake_ra
    curmocktable['DEC_fake'] = fake_dec

    # Core columns to keep at the beginning
    fixed_cols = ['name', 'logg', 'Teff', 'mh', 'cto', 'logkzz', 'PH3', 'Radius_pc', 'SpT']

    # Interleave f_ and e_ columns for each band
    bands = parameters['bands']  # e.g., ['F098M', 'F105W', ..., 'F606W']
    band_cols = [col for band in bands for col in (f'f_{band}', f'e_{band}')]

    # Combine into the new column order
    new_order = fixed_cols + ['RA_fake', 'DEC_fake'] + band_cols

    # Filter out columns that don't exist in the table
    available_cols = [col for col in new_order if col in curmocktable.colnames]

    # Apply column order safely
    curmocktable = curmocktable[available_cols]
    
    #generating a new set of magnitudes for each band
    for band in parameters['bands']:
        # Convert mags to flux (in cgs units)
        flux_ori_cgs = 10**((mag_ori[f'f_{band}'] + 48.6) * - 0.4)

        #start the main program
        #make simulated catalogs at different detection band magnitude
        flux_detec_theory = 10**((m_detec_set + 48.6 )* -0.4) #cgs

        # Choose correct original flux for scaling factor
        flux_ori_detec = 10 ** ((mag_ori[f"f_{parameters['detection_band']}"] + 48.6) * -0.4)
        factor_theory = flux_detec_theory/flux_ori_detec #cgs

        # Scale the original fluxes
        flux_theory_band = flux_ori_cgs * factor_theory #array of nstars in cgs

        # Ensure error column exists
        e_colname = 'e_%s' % band
        if e_colname not in curmocktable.colnames:
            curmocktable[e_colname] = np.zeros(len(curmocktable)) 

        # Set the scaled flux in the current band
        curmocktable['f_%s'%band] = np.zeros_like(flux_theory_band)
    
        # Load constants
        exptime = parameters['EXPTIME'][band]
        flam = parameters['PHOTFLAM'][band]
        plam = parameters['PHOTPLAM'][band]
        limmag_5s = parameters['depths'][band]
        limmag_1s = limmag_5s + (2.5 *np.log10(5)) # 5 sigma to 1 sigma magnitude 

        # Convert limiting magnitude to CGS flux
        limflux_cgs = 10**((limmag_1s + 48.6) / -2.5)  # flux_nu in cgs
        lim_nelectron = (c_AAs * limflux_cgs / (flam * plam**2)) * exptime  # electrons

        # Convert theoretical flux to electron units
        flux_theory_band_es = c_AAs * flux_theory_band / (flam * plam**2)  # e/s
        flux_theory_band_electron = flux_theory_band_es * exptime  # electrons

        # Compute error and SNR
        e_nelectron = np.sqrt(lim_nelectron**2 + flux_theory_band_electron)  # total noise
        snr_electron = flux_theory_band_electron / e_nelectron
        noise = flux_theory_band / snr_electron  # noise in CGS

        # Add noise to flux
        flux_noise = flux_theory_band + np.random.normal(loc=0, scale=noise)
        # Store to curmocktable
        curmocktable[f'f_{band}'] = flux_noise
        curmocktable[e_colname] = noise

    ############### reorder and save the columns of table ###############

    # Core columns to keep at the beginning
    fixed_cols = ['name', 'logg', 'Teff', 'mh', 'cto', 'logkzz', 'PH3', 'Radius_pc', 'SpT']

    # Interleave f_ and e_ columns for each band
    bands = parameters['bands']  # e.g., ['F098M', 'F105W', ..., 'F606W']
    band_cols = [col for band in bands for col in (f'f_{band}', f'e_{band}')]

    # Combine into the new column order
    new_order = fixed_cols + ['RA_fake', 'DEC_fake'] + band_cols

    # Filter out columns that don't exist in the table
    available_cols = [col for col in new_order if col in curmocktable.colnames]

    # Apply column order safely
    curmocktable = curmocktable[available_cols]
    
    # convert flux_cgs to flux at zp 25
    flux25_table = curmocktable.copy()
    for curband in band_cols:
        flux25_table[curband] = 10**(-0.4 * (((-2.5 * np.log10(flux25_table[curband])) - 48.6 - 25)))
    save_mock = os.path.join(output_path, f"{parameters['fields_name']}/mag_{m_detec_set}", f"{mh}_flux25_move_to_mag{m_detec_set}.dat")
    os.makedirs(os.path.dirname(save_mock), exist_ok=True)

    ascii.write(flux25_table, save_mock, overwrite=True, delimiter="\t")

def run_eazy(parameters, m_detec_set, outputfolder, mh, template,
             mainoutputfile_prefix=None):
    import eazy

    eazy_folder = outputfolder+'/Eazy' 
    if not os.path.exists(eazy_folder):
        os.makedirs(eazy_folder, exist_ok=True)
    
    if not os.path.exists('templates'):
        eazy.symlink_eazy_inputs() 

    if mainoutputfile_prefix is None:
        mainoutputfile_prefix = parameters['fields_name']
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    params = {}

    mainoutputfile = f"{eazy_folder}/{mainoutputfile_prefix}_{template}"
    ez_inputfile = eazy_folder+'/%s_%s.FITS'%(parameters['fields_name'], template)
    print('running eazy to make %s.h5'%mainoutputfile)
    readtable = ascii.read(outputfolder+'/%s_flux25_move_to_mag%s.dat'%(mh, m_detec_set))
    outtable = Table()
    outtable['id'] = np.char.replace(readtable['name'], '.nc', '')
    outtable['ra'] = readtable['RA_fake']
    outtable['dec'] = readtable['DEC_fake']

    # Assign z_spec (only if template is 'spex')
    if template.lower() == 'spex':
        outtable['z_spec'] = np.zeros_like(len(readtable))+1e-7

    # Process each band
    for band in parameters['bands']:
    #for band in filters:
        f_col = f'f_{band}'
        e_col = f'e_{band}'

        flux = readtable[f_col]
        err = readtable[e_col]

        # Replace negative fluxes with 0.01 * flux limit
        # Compute 1σ limiting flux from depth and zeropoint
        limmag_5s = parameters['depths'][band]
        limmag_1s = limmag_5s + (2.5 *np.log10(5)) # 5 sigma to 1 sigma magnitude 
        limflux = 10 ** (-0.4 * (limmag_1s - 25)) #zp=25

        # Replace negative fluxes
        mask = (flux < 0) | np.isnan(flux)
        corrected_flux = np.where(mask, 0.01 * limflux, flux)
        corrected_errflux = np.where(mask, limflux, err)

        # Update output table
        outtable[f_col] = corrected_flux
        outtable[e_col] = corrected_errflux

    outtable.write(ez_inputfile,overwrite=True)
    print(template)

    if template.lower() == 'v1p3':
        params['TEMPLATES_FILE'] = 'templates/eazy_v1.3.spectra.param'
    elif template.lower() == 'v1p0':
        params['TEMPLATES_FILE'] = 'templates/eazy_v1.0.spectra.param'
    elif template.lower() == 'larson':
        params['TEMPLATES_FILE'] = 'templates/LarsonSEDTemplates/Larson.param'
    elif template.lower() == 'spex':
        params['TEMPLATES_FILE'] = 'templates/spex_prism_all/dwarf_list.param'
    elif template.lower() == 'hainline':
        params['TEMPLATES_FILE'] = 'templates/Hainline2023/Hainline2023.param'
        
    params['CATALOG_FILE'] = ez_inputfile
    params['N_MIN_COLORS'] = 2
        
    params['OUTPUT_DIRECTORY'] = eazy_folder
    params['MAIN_OUTPUT_FILE'] = mainoutputfile
    
    params['PRIOR_FILTER'] = 205 #HST: F160W=205, JWST: F444W=377, Roman: F158=429
    params['PRIOR_ABZP'] = 25
    params['PRIOR_FILE'] = '/Users/onnalininnala/Intern/prior_flat_extended_JWST.dat'
    
    if template.lower() =='spex':
        params['FIX_ZSPEC'] = 'y'
        params['Z_MIN'] = -0.005
        params['Z_MAX'] = 0.05
        params['Z_STEP'] = 0.001
        params['APPLY PRIOR'] = 'y'
        params['Z_COLUMN'] = 'z_p'
        params['TEMPLATE_COMBOS'] = 1
    else:
        params['FIX_ZSPEC'] = 'n'
        params['Z_MIN'] = 0.1
        params['Z_MAX'] = 20.00
        params['Z_STEP'] = 0.01
        params['Z_COLUMN'] = 'z_p'
        params['TEMPLATE_COMBOS'] = 99
    params['SYS_ERR'] = 0.02

    params['Z_STEP_TYPE'] = 1
    params['REST_FILTERS'] = '41,45,49,64'
    params['TEMP_ERR_A2'] = 0
        
    params['VERBOSITY'] = 1
    params['APPLY_IGM'] = 'n'

        #initialize PhotoZ object
    ez = eazy.photoz.PhotoZ(param_file=None,
                            translate_file='/Users/onnalininnala/Intern/zphot.translate',
                            zeropoint_file=None, params=params,
                            load_prior=True, load_products=False)
    
    #fit catalog
    ez.fit_catalog(n_proc=8)
        
    #derived parameters (z params, RF colors, masses, SFR, etc.)
    zout = ez.standard_output(prior=True, cosmology=cosmo, extra_rf_filters=[], absmag_filters =[], save_fits=0)
    zout_table = zout[0]
    pct = parameters['eazy_pct']
    z_user = ez.pz_percentiles([pct])
    z_user_flat = z_user.flatten()
    tab_absmag = ez.abs_mag(f_numbers=[433,140,141],
                                rest_kwargs={'percentiles': [16,50,84]})  
    
    # Add new column to zout table
    zout_table.add_column(Column(z_user_flat, name=f"z_pct_{pct}"))
    zout_table.add_column(Column(tab_absmag))

    # Replace zout with updated version
    zout = (zout_table, zout[1])
    filename = f"{mainoutputfile}.zout.fits"
    # Save only the table part
    zout_table.write(filename, format='fits', overwrite=True)
    
    return ez, zout_table

def eazy(parameters, m_detec_set, outputfolder, mh):
    eazy_fol = outputfolder+'/Eazy'
    if not os.path.exists(eazy_fol):
        os.makedirs(eazy_fol, exist_ok=True)

    print(parameters['templates'][0],parameters['templates'][1])
    ez_1st,zout = run_eazy(parameters, m_detec_set, outputfolder, mh, template=parameters['templates'][0], 
                mainoutputfile_prefix=None) #galaxy
    print(ez_1st)
    ez_2nd,_ = run_eazy(parameters, m_detec_set, outputfolder, mh, template=parameters['templates'][1], 
                mainoutputfile_prefix=None) #stellar
    ### Generate the summary table section.
    # Store EAZY chi² stats
    obj_ids = ez_1st.OBJID
    ez_chi2_galaxy = (ez_1st.chi2_best)
    ez_chi2_spex = (ez_2nd.chi2_best)
    ez_zbest_galaxy = (ez_1st.zbest)
    pct = f"z_pct_{parameters['eazy_pct']}"
    z_pct = zout[pct]

    # Create result table
    result_table = Table(names=('ID','Z_best', 'eazy_chi2_galaxy', 'eazy_chi2_spex',
                                'eazy_best_fit_chi2', f"{pct}"),
                        dtype=('S50', 'f8', 'f8', 'f8', 'S10', 'f8'))

    for curid in range(len(ez_1st.OBJID)):
        # Fill result table row-by-row
        if curid < len(ez_chi2_spex) and curid < len(ez_chi2_galaxy):
            ez_best_fit = parameters['templates'][1] if ez_chi2_spex[curid] < ez_chi2_galaxy[curid] else parameters['templates'][0]
        else:
            ez_best_fit = None

        result_table.add_row((obj_ids[curid], ez_zbest_galaxy[curid], ez_chi2_galaxy[curid], ez_chi2_spex[curid], 
                                ez_best_fit, z_pct[curid]))
    # Save the result table
    ascii.write(result_table, f"{eazy_fol}/{mh}_bestfit_{m_detec_set}.dat", overwrite=True, delimiter='\t')

    return eazy_fol

def color_borsani(parameters):
    """
    Defines color selection criteria based on Borsani 2022, for specific filter sets.

    Parameters:
        parameters (dict): must include 'field' key with value 'set_b' = BoRG field.

    Returns:
        tuple of dicts: (criteria_z8_098, criteria_z8_105, criteria_z9_105)
    """
    # Selection is based on a linear color-color cut: y_coef * (y1 - y2) > slope * (x1 - x2) + y_int
    # 'add_cri' is a dictionary containing additional selection conditions.
    # Each key represents a condition (e.g. SNR in a specific filter),
    # and the value is a list: [threshold value, comparison direction]
    # where:
    #   - threshold: the required value for that condition
    #   - direction: 1 means value must be greater than the threshold (>)
    #                0 means value must be less than the threshold (<)
    filter_range = ['F300X', 'F475X', 'F350LP', 'F435W','F475W', 'F475X', 'F555W','F600LP', 'F606W', 'F625W',
                    'F775W', 'F814W', 'F850LP', 'F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']
    if parameters['criteria_set'].lower() == 'set_b':
        criteria_z8_098 ={'name': 'z8_098',
                        'x_filter':['F125W', 'F160W', 0.5],
                        'y_filter':['F098M', 'F125W', 1.75],
                        'y_coef':0.15,
                        'slope':1,
                        'y_int':0.2425,
                        'add_cri': {'SNR_F140W': ['F125W', 6.0, 'gt'],
                                    'SNR_F160W': ['F160W', 4.0, 'gt'],
                                    'Non-Detec': ['F814W', 1.0, 'lt', filter_range]
                                    }
                            }
        criteria_z8_105 ={'name': 'z8_105',
                        'x_filter':['F125W', 'F160W', 0.5],
                        'y_filter':['F105W', 'F125W', 0.45],
                        'y_coef':1,
                        'slope':1.5,
                        'y_int':0.45,
                        'add_cri': {'SNR_F140W': ['F125W', 6.0, 'gt'],
                                    'SNR_F160W': ['F160W', 4.0, 'gt'],
                                    'Non-Detec': ['F814W', 1.0, 'lt', filter_range]
                                    }
                            }
        criteria_z9_105 ={'name': 'z9_105',
                        'x_filter':['F140W', 'F160W', 0.3],
                        'y_filter':['F105W', 'F140W', 1.5],
                        'y_coef':1,
                        'slope':5.33,
                        'y_int':0.7,
                        'add_cri': {'SNR_F140W': ['F140W', 6.0, 'gt'],
                                    'SNR_F160W': ['F160W', 4.0, 'gt'],
                                    'Non-Detec': ['F814W', 1.0, 'lt', filter_range]
                                    }
                            }
        return criteria_z8_098, criteria_z8_105, criteria_z9_105

    else:
        raise ValueError(f"Invalid field '{parameters.get('criteria_set')}'. Expected 'set_b'.")

def color_bouwen(parameters):
    #this criteria from Bouwens et al. 2025 DOI: 10.1088/0004-637X/803/1/34
    # Selection is based on a linear color-color cut: y_coef * (y1 - y2) > slope * (x1 - x2) + y_int
    # 'add_cri' is a dictionary containing additional selection conditions.
    # Each key represents a condition (e.g. SNR in a specific filter),
    # and the value is a list: [threshold value, comparison direction]
    # where:
    #   - threshold: the required value for that condition
    #   - direction: 1 means value must be greater than the threshold (>)
    #                0 means value must be less than the threshold (<)
    #
    #if you don't use elf olw syntesis model, you must add 'F435W' in list_chi7 and list_chi8
    # and 'add_cri': {'SNR_F435W': ['F435W', 2.0, 0]} in z5-z8_GS_GN, z5-z8_BoRG,
    all_condi = ['F105W', 'F125W', 'F140W', 'F160W']
    list_chi7 = ['F606W']
    list_chi8 = ['F606W','F814W']
    list_chi10 = ['F606W','F814W']
    filter_range = ['F606W','F814W', 'F850LP', 'F105W','F125W', 'F140W', 'F160W']

    if parameters['criteria_set'].lower() == 'set_a':
        z5_GS_GN = {'name': 'z5_GS_GN',
                        'x_filter':['F850LP', 'F160W', 1.2],
                        'y_filter':['F606W', 'F775W', 1.75],
                        'y_coef':1,
                        'slope':1.6,
                        'y_int':1,  
                        'add_cri': {'all_condi': [24, 'gt', all_condi]
                                    }   
                    }
        z6_GS_GN = {'name': 'z6_GS_GN',
                        'x_filter':['F105W', 'F160W', 1.0],
                        'y_filter':['F775W', 'F850LP', 1.0],
                        'y_coef':1,
                        'slope':0.78,
                        'y_int':1.0, 
                        'add_cri': {'two_filter':['F606W', 'F850LP', 2.7, 'gt'],
                                    'SNR_F606W': ['F606W',2.0, 'gt'],
                                    'all_condi': [24, 'gt', all_condi]
                                    }     
                    }
        z7_GS_GN = {'name': 'z7_GS_GN',
                        'x_filter':['F125W', 'F160W', 0.45],
                        'y_filter':['F850LP', 'F105W', 0.7],
                        'y_coef':1,
                        'slope':0.8,
                        'y_int':0.7,    
                        'add_cri': {'SNR_F814W': ['F814W', 1.5, 'lt'],
                                    'chi2_opt': [3.0, 'lt', list_chi7],
                                    'two_filter':['F814W', 'F125W', 1.0, 'gt'],
                                    'Non-Detec': ['F814W', 2.0, 'lt', filter_range], 
                                    'all_condi': [24, 'gt', all_condi]
                                    }  
                    }
        z8_GS_GN = {'name': 'z8_GS_GN',
                        'x_filter':['F125W', 'F160W', 0.5],
                        'y_filter':['F105W', 'F125W', 0.45],
                        'y_coef':1,
                        'slope':0.75,
                        'y_int':0.525,   
                        'add_cri': {'chi2_opt': [3.0, 'lt', list_chi8],
                                    'two_filter':['F814W', 'F125W', 1.0, 'gt'],
                                    'Non-Detec': ['F850LP', 2.0, 'lt', filter_range], #814 and bluer (if avialable)
                                    'all_condi': [24, 'gt', all_condi]
                                    }    
                    }
    if parameters['criteria_set'].lower() == 'set_b':    
        z4_BoRG = {'name': 'z4_BoRG',
                        'x_filter':['F775W', 'F125W', 1.0],
                        'y_filter':['F435W', 'F606W', 1.0],
                        'y_coef':1,
                        'slope':1.6,
                        'y_int':1,
                        'add_cri': {'all_condi': [25, 'gt', all_condi]
                                    }       
                    }
        z5_BoRG = {'name': 'z5_BoRG',
                        'x_filter':['F850LP', 'F160W', 1.2],
                        'y_filter':['F606W', 'F775W', 1.75],
                        'y_coef':1,
                        'slope':0.8,
                        'y_int':1.2,  
                        'add_cri': {'all_condi': [25, 'gt', all_condi],
                                    'Non-Detec': ['F475W', 2.0, 'lt', filter_range] #435 or bluer
                                    }   
                    }
        z6_BoRG = {'name': 'z6_BoRG',
                        'x_filter':['F098M', 'F160W', 1.0],
                        'y_filter':['F775W', 'F850LP', 1.0],
                        'y_coef':1,
                        'slope':0.6,
                        'y_int':1.0, 
                        'add_cri': {'two_filter':['F606W', 'F850LP', 2.7, 'gt'],
                                    'SNR_F606W': [2.0, 'gt'],
                                    'Non-Detec': ['F475W', 2.0, 'lt', filter_range], #435 or bluer
                                    'all_condi': [25, 'gt', all_condi]
                                    }     
                    }
        z7_BoRG = {'name': 'z7_BoRG',
                        'x_filter':['F125W', 'F160W', 0.5],
                        'y_filter':['F850LP', 'F125W', 1.3],
                        'y_coef':1,
                        'slope':0.8,
                        'y_int':0.7, 
                        'add_cri': {'SNR_F814W': ['F814W', 1.5, 'lt'],
                                    'chi2_opt': [3.0, 'lt', list_chi7],
                                    'two_filter':['F814W', 'F125W', 1.0, 'gt'],
                                    'Non-Detec': ['F814W', 2.0, 'lt', filter_range], #775 and bluer (if avialable)
                                    'all_condi': [25, 'gt', all_condi]
                                    }     
                    }
        z8_BoRG = {'name': 'z8_BoRG',
                        'x_filter':['F125W', 'F160W', 0.5],
                        'y_filter':['F098M', 'F125W', 1.3],
                        'y_coef':1,
                        'slope':0.75,
                        'y_int':1.3, 
                        'add_cri': {'chi2_opt': [3.0, 'lt', list_chi8],
                                    'two_filter':['F814W', 'F125W', 1.0, 'gt'],
                                    'Non-Detec': ['F850LP', 2.0, 'lt', filter_range], #814 and bluer (if avialable)
                                    'all_condi': [25, 'gt', all_condi]
                                    }     
                    }
    
    if parameters['criteria_set'].lower() == 'set_c':
        z5_Cosmos_EGS = {'name': 'z5_Cosmos',
                        'x_filter':['F814W', 'F160W', 1.25],
                        'y_filter':['F606W', 'F814W', 1.3],
                        'y_coef':1,
                        'slope':0.72,
                        'y_int':1.3,  
                        'add_cri': {'all_condi': [25, 'gt', all_condi]
                                    }   
                    }
        z6_Cosmos_EGS = {'name': 'z6_Cosmos',
                        'x_filter':['F125W', 'F160W', 0.4],
                        'y_filter':['F814W', 'F125W', 0.8],
                        'y_coef':1,
                        'slope':2,
                        'y_int':0.8,  
                        'add_cri': {'all_condi': [25, 'gt', all_condi]
                                    }     
                    }
        z7_Cosmos_EGS = {'name': 'z7_Cosmos',
                        'x_filter':['F125W', 'F160W', 0.4],
                        'y_filter':['F814W', 'F125W', 2.2],
                        'y_coef':1,
                        'slope':2,
                        'y_int':2.2, 
                        'add_cri': {'all_condi': [25, 'gt', all_condi]
                                    }      
                    }
        z8_Cosmos_EGS = {'name': 'z8_Cosmos',
                        'x_filter':['F125W', 'F160W', 0.4],
                        'y_filter':['F814W', 'F125W', 2.2],
                        'y_coef':1,
                        'slope':2,
                        'y_int':2.2,    
                        'add_cri': {'all_condi': [25, 'gt', all_condi]
                                    }   
                    }
        
        z10_Cosmos_EGS = {'name': 'z10_Cosmos',
                        'x_filter':['F160W', 3.6, 1.4],
                        'y_filter':['F125W', 'F160W', 1.2],
                        'y_coef':1,
                        'slope':2,
                        'y_int':2.2,    
                        'add_cri': {'all_condi': [25, 'gt', all_condi],
                                    'chi2_opt': [2, 'lt', list_chi10]
                                    }   
                    }

    if parameters['criteria_set'].lower() == 'set_a':
        return z6_GS_GN, z7_GS_GN, z8_GS_GN
    elif parameters['criteria_set'].lower() == 'set_b':
        return z4_BoRG, z5_BoRG, z6_BoRG, z7_BoRG, z8_BoRG
    elif parameters['criteria_set'].lower() == 'set_c':
        return z5_Cosmos_EGS, z6_Cosmos_EGS, z7_Cosmos_EGS, z8_Cosmos_EGS, z10_Cosmos_EGS
    else:
        raise ValueError(f"Unrecognized field value: {parameters['criteria_set']}")
    
def color_selection(specific_cri_z, parameters_file, flux_path, m_detec_set, output_folder, type, log_file_path, photoz_path=None):   
    output_table = Table()
    
    color_folder = output_folder+'/color'       
    if not os.path.exists(color_folder):
        os.makedirs(color_folder, exist_ok=True)

    def normalize_direction(direction):
        """Ensure direction is always 'gt' or 'lt'."""
        if isinstance(direction, str):
            return direction.lower()
        if isinstance(direction, (int, bool)):
            return 'gt' if direction else 'lt'
        raise ValueError(f"Invalid direction: {direction}")

    def extract_filters(data):
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        filters = set()

        # Pattern matches F followed by 4 digits, and optionally ends in W, M, X, or LP
        pattern = re.compile(r'^F\d{3}(W|M|LP|X)?$')

        # From x_filter and y_filter
        for key in ['x_filter', 'y_filter']:
            for item in data.get(key, []):
                if isinstance(item, str) and pattern.match(item):
                    filters.add(item)

        # From add_cri
        add_cri = data.get('add_cri', {})
        for key, value in add_cri.items():
            if key in ('Non-Detec', 'all_condi'):
                continue  # skip these keys entirely
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and pattern.match(item):
                    filters.add(item)
                elif isinstance(item, list):
                    for subitem in item:
                        if isinstance(subitem, str) and pattern.match(subitem):
                            filters.add(subitem)
        return sorted(filters)

    def chi2_opt(flux_table, filters):
        """
        Compute signed chi^2 summed over filters.
        """
        chi2 = np.zeros(len(flux_table))

        for filt in filters:
            f_col = f"f_{filt}"
            e_col = f"e_{filt}"

            if f_col not in flux_table.colnames or e_col not in flux_table.colnames:
                continue

            flux = np.asarray(flux_table[f_col])
            err = np.asarray(flux_table[e_col])

            mask = err > 0
            chi2[mask] += np.sign(flux[mask]) * (flux[mask] / err[mask]) ** 2

        return chi2

    def additional_condition(add_cond, mag_table, flux_table, eazy_photoz_table, all_fil):
        final_mask = np.ones(len(mag_table), dtype=bool)

        for key, val in add_cond.items():
            if key.startswith("SNR_"):
                filt, threshold, direction = val
                direction = normalize_direction(direction)
                col = f'SNR_{filt}'
                if col not in mag_table.colnames:
                    raise KeyError(f"{col} not found in mag_table columns")
                final_mask &= (mag_table[col] > threshold) if direction == 'gt' else (mag_table[col] < threshold)

            elif key.lower() == "chi2_opt":
                threshold, direction, filters = val
                direction = normalize_direction(direction)
                chi = chi2_opt(flux_table, filters)
                final_mask &= (chi > threshold) if direction == 'gt' else (chi < threshold)

            elif key.lower() == "bluer105":
                bluer_105, threshold, direction = val
                direction = normalize_direction(direction)
                for iband in all_fil:
                    if iband in bluer_105:
                        col = f'SNR_{iband}'
                        if col not in mag_table.colnames:
                            raise KeyError(f"{col} not found in mag_table columns")
                        final_mask &= (mag_table[col] > threshold) if direction == 'gt' else (mag_table[col] < threshold)

            elif key.lower() == "all_condi":
                threshold, direction, filter_all = val
                direction = normalize_direction(direction)
                filters_avi = all_fil
                threshold_filter = threshold * len(filters_avi) / len(filter_all)
                chi = chi2_opt(flux_table, filters_avi)
                final_mask &= (chi > threshold_filter) if direction == 'gt' else (chi < threshold_filter)

            elif key.lower() == "two_filter":
                f1, f2, threshold, direction = val
                direction = normalize_direction(direction)
                col1, col2 = f'f_{f1}', f'f_{f2}'
                if col1 in mag_table.colnames and col2 in mag_table.colnames:
                    final_mask &= ((mag_table[col1] - mag_table[col2]) > threshold) if direction == 'gt' else ((mag_table[col1] - mag_table[col2]) < threshold)

            elif key.lower() == "abmag":
                filt, threshold, direction = val
                direction = normalize_direction(direction)
                col = f'f_{filt}'
                if col in mag_table.colnames:
                    final_mask &= (mag_table[col] > threshold) if direction == 'gt' else (mag_table[col] < threshold)

            elif key.lower() == "non-detec":
                target, threshold, direction, filter_range = val
                direction = normalize_direction(direction)
                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                target_index = filter_range.index(target)
                before_filters = [f for f in all_fil if filter_range.index(f) < target_index]
                for iband in before_filters:
                    col = f'SNR_{iband}'
                    if col not in mag_table.colnames:
                        raise KeyError(f"{col} not found in mag_table columns")
                    final_mask &= (mag_table[col] > threshold) if direction == 'gt' else (mag_table[col] < threshold)

        return final_mask
    
    def additional_condition_z6(add_cond, mag_table, flux_table, all_fil):

        mask_non_detec = np.ones(len(mag_table), dtype=bool)
        mask_twofil = np.zeros(len(mag_table), dtype=bool)
        mask_snr = np.zeros(len(mag_table), dtype=bool)
        mask_all_condi = np.ones(len(mag_table), dtype=bool)

        for key, val in add_cond.items():

            # -------------------------------------------------
            # SNR condition
            # -------------------------------------------------
            if key.startswith("SNR_"):

                filt, threshold, direction = val
                direction = normalize_direction(direction)

                col = f'SNR_{filt}'

                if col not in mag_table.colnames:
                    raise KeyError(f"{col} not found in mag_table columns")

                if direction == 'gt':
                    mask_snr |= (mag_table[col] > threshold)
                else:
                    mask_snr |= (mag_table[col] < threshold)

            # -------------------------------------------------
            # Two-filter color condition
            # -------------------------------------------------
            elif key.lower() == "two_filter":

                f1, f2, threshold, direction = val
                direction = normalize_direction(direction)

                col1 = f'f_{f1}'
                col2 = f'f_{f2}'

                if col1 in mag_table.colnames and col2 in mag_table.colnames:

                    if direction == 'gt':
                        mask_twofil |= (
                            (mag_table[col1] - mag_table[col2]) > threshold
                        )
                    else:
                        mask_twofil |= (
                            (mag_table[col1] - mag_table[col2]) < threshold
                        )

            # -------------------------------------------------
            # all_condi
            # -------------------------------------------------
            elif key.lower() == "all_condi":

                threshold, direction, filter_all = val
                direction = normalize_direction(direction)

                filters_avi = all_fil

                threshold_filter = (
                    threshold * len(filters_avi) / len(filter_all)
                )

                chi = chi2_opt(flux_table, filters_avi)

                if direction == 'gt':
                    mask_all_condi &= (chi > threshold_filter)
                else:
                    mask_all_condi &= (chi < threshold_filter)

            # -------------------------------------------------
            # Non-detection condition
            # -------------------------------------------------
            elif key.lower() == "non-detec":

                target, threshold, direction, filter_range = val

                direction = normalize_direction(direction)

                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                if target not in filter_range:
                    raise ValueError(
                        f"{target} not found in filter_range: {filter_range}"
                    )

                target_index = filter_range.index(target)

                before_filters = [
                    f for f in all_fil
                    if filter_range.index(f) < target_index
                ]

                for iband in before_filters:

                    col = f'SNR_{iband}'

                    if col not in mag_table.colnames:
                        raise KeyError(
                            f"{col} not found in mag_table columns"
                        )

                    if direction == 'gt':
                        mask_non_detec &= (
                            mag_table[col] > threshold
                        )
                    else:
                        mask_non_detec &= (
                            mag_table[col] < threshold
                        )

        # -------------------------------------------------
        # Final combination
        # -------------------------------------------------
        final_mask = (
            mask_non_detec
            & (mask_twofil | mask_snr)
            & mask_all_condi
        )

        return final_mask
    
    # If photoz_path is required but not provided, raise an error (optional)
    if photoz_path is None:
        raise ValueError("photoz_path is required for color criteria.")

    # Load photo-z data if available
    if photoz_path:
        try:
            photoz_table = ascii.read(photoz_path)
        except Exception as e:
            print(f"Warning: Could not read photoz file: {photoz_path}")
            print(e)
            photoz_table = None
    else:
        photoz_table = None
    
    # Load the flux table
    flux_25 = ascii.read(flux_path)
    # Add the 'ID' column from flux_25 
    output_table.add_column(flux_25['name'], name='ID') 


    with open(log_file_path, 'a') as log_file:
    # Helper function to log to both console and file
        def log(msg):
            print(msg)
            log_file.write(str(msg) + '\n')

        log(f"Starting program for magnitude {m_detec_set}")
        log(f"Stellar model is {type} type \n")

        for cur_cri_z in specific_cri_z: 
            # Get all 'f_' prefixed columns in the catalog
            catalog_cols = [col for col in flux_25.colnames if col.startswith('f_')]
            catalog_filters = [col[2:] if col.startswith('f_') else col for col in catalog_cols]

            #Check criteria required filers
            # Extract all required filters from cur_cri_z except filters in conditions applied with all criteria
            required_filters = extract_filters(cur_cri_z)

            all_fil = []

            add_cri = cur_cri_z.get('add_cri', {})

            # Safe access Non-Detec if present
            non_detec = add_cri.get('Non-Detec')
            if non_detec and len(non_detec) == 4:
                target, _, _, filter_range = non_detec
                if target in filter_range:
                    if target not in filter_range:
                        raise ValueError(
                            f"{target} not found in filter_range: {filter_range}"
                        )

                    target_index = filter_range.index(target)
                    before_filters = [f for f in catalog_filters if f in filter_range and filter_range.index(f) < target_index]
                    filters_to_remove = before_filters
                else:
                    filters_to_remove = []
            else:
                filters_to_remove = []

            color_criteria = parameters_file.get('color_criteria', '')
            color_set = parameters_file.get('criteria_set', '')

            if color_criteria.lower() == 'borsani_2022':
                all_fil = filters_to_remove
            elif color_criteria.lower() == 'bouwens_2015':
                all_fil_nondetec = filters_to_remove
                all_condi = add_cri.get('all_condi', [None, None, []])
                if len(all_condi) > 2:
                    all_fil = set(all_condi[2]) | set(all_fil_nondetec)  # union
                else:
                    all_fil = set(all_fil_nondetec)

            # Keep only filters present in catalog_filters
            required_all_con = [f for f in all_fil if f in catalog_filters]

            # Combine filters while avoiding duplicates
            combined_required = required_filters + [f for f in required_all_con if f not in required_filters]

            # Identify missing flux columns
            missing_flux_cols = [col for col in combined_required if col not in catalog_filters]
            log(f"{cur_cri_z['name']} in {parameters_file['color_criteria']}")
            log(f"Available catalog columns: {catalog_filters}")
            log(f"Flux columns required for selection criteria: {combined_required}")
            # Check if any are missing
            if missing_flux_cols:
                log(f"→ Missing: {missing_flux_cols}\n")
                continue  # Skip this iteration
                
            mag_table = Table()
            # Calculate magnitudes and SNRs
            for filt in catalog_filters:
                valid = (flux_25[f'f_{filt}'] > 0) & (flux_25[f'e_{filt}'] > 0)
                mag = np.where(valid, -2.5 * np.log10(flux_25[f'f_{filt}']) + 25, parameters_file['depths'][filt]) #flux_zp25 to ABmag
                snr = np.where(valid, flux_25[f'f_{filt}'] / flux_25[f'e_{filt}'], 1.0) #SNR from flux_zp25
                mag_table[f'f_{filt}'] = mag
                mag_table[f'SNR_{filt}'] = snr

                # calculate error flux by propagation of uncertainty
                #function = a*log10(bA); 𝜎_f ≈ abs( (-2.5 * 𝜎_A) / (A * ln(10)))
                #mag_table[f'e_{filt}'] = np.abs( ( -2.5 * flux_25[f'e_{filt}'] ) / (flux_25[f'f_{filt}'] * np.log(10)))
                #mag_table[f'SNR_{filt}'] = mag_table[f'f_{filt}'] / mag_table[f'e_{filt}']

            # Compute color axes
            if cur_cri_z['name'] == 'z10_Cosmos':
                x = mag_table[f'f_{cur_cri_z["x_filter"][0]}'] - cur_cri_z["x_filter"][1]
            else:
                x = mag_table[f'f_{cur_cri_z["x_filter"][0]}'] - mag_table[f'f_{cur_cri_z["x_filter"][1]}']
                
            y = mag_table[f'f_{cur_cri_z["y_filter"][0]}'] - mag_table[f'f_{cur_cri_z["y_filter"][1]}']

            # Core color criteria
            cond1 = x < cur_cri_z['x_filter'][2]
            cond2 = y > cur_cri_z['y_filter'][2]
            cond3 = (cur_cri_z['y_coef'] * y) > (cur_cri_z['slope'] * x + cur_cri_z['y_int'])

            # Combine main conditions
            final_mask = cond1 & cond2 & cond3
            
            if 'add_cri' in cur_cri_z and cur_cri_z['add_cri']:
                if cur_cri_z['name'] != 'z6_GS_GN':
                    final_mask &= additional_condition(
                        cur_cri_z['add_cri'], mag_table, flux_25, photoz_table, all_fil
                    )
                else:
                    final_mask &= additional_condition_z6(
                        cur_cri_z['add_cri'], mag_table, flux_25, all_fil
                    )

            if color_criteria.lower() == 'borsani_2022':
                #Photometric redshift estimation criteria
                pct = f"z_pct_{parameters_file['eazy_pct']}"
                if cur_cri_z['name'].startswith('z8'):
                    z_range = [7.5,8.5]
                    z_prob = photoz_table[f"{pct}"] > 6.5

                elif cur_cri_z['name'].startswith('z9'):
                    z_range = [8.5,9.5]
                    z_prob = photoz_table[f"{pct}"] > 7.5

                final_mask &= (photoz_table['Z_best'] >= z_range[0]) & (photoz_table['Z_best'] <= z_range[1]) & z_prob

            elif color_criteria.lower() == 'bouwens_2015' and color_set == 'set_c':
                gal_template = 'hainline'
                pct = f"z_pct_{parameters_file['pct']}"
                #Photometric redshift estimation criteria
                if cur_cri_z['name'].startswith('z5'):
                    z_range = [4.2,5.5]

                elif cur_cri_z['name'].startswith('z6'):
                    z_range = [5.5,6.3]

                elif cur_cri_z['name'].startswith('z7'):
                    z_range = [6.3,7.3]

                elif cur_cri_z['name'].startswith('z8'):
                    z_range = [7.3,9.0]
                
                elif cur_cri_z['name'].startswith('z10'):
                    z_range = [9.5,10.5]
                    
                final_mask &= (photoz_table['Z_best'] >= z_range[0]) & (photoz_table['Z_best'] <= z_range[1]) & (photoz_table["eazy_best_fit_chi2"] == gal_template) 

            # Output
            num_pass = np.sum(final_mask)
            # Create a new Column with the True/False values
            mask_column = Column(final_mask, name=f"{cur_cri_z['name']}")

            # Add it to the table
            output_table.add_column(mask_column)
            log(f"Number of targets that pass all selection criteria in {cur_cri_z['name']}: {num_pass}\n")

        # Save output inside with-block if you want to log about it
        output_filename = f"{color_folder}/{type}_{parameters_file['color_criteria']}.dat"
        ascii.write(output_table, output_filename, overwrite=True, delimiter='\t')

def color_selection_basic(cur_cri_z,  flux_path):    
    #### Loading flux zp 25 — converting to AB magnitude scale.
    flux_25 = ascii.read(flux_path)

    # Extract all required filters
    list_filter = list(set(cur_cri_z['x_filter'][:2] + cur_cri_z['y_filter'][:2]))
    mag_table = Table()

    # Calculate magnitudes and SNRs
    for filt in list_filter:
        mag_table[f'f_{filt}'] = -2.5 * np.log10(flux_25[f'f_{filt}']) + 25

    # Compute color axes
    x = mag_table[f'f_{cur_cri_z["x_filter"][0]}'] - mag_table[f'f_{cur_cri_z["x_filter"][1]}']
    y = mag_table[f'f_{cur_cri_z["y_filter"][0]}'] - mag_table[f'f_{cur_cri_z["y_filter"][1]}']

    # Core color criteria
    cond1 = x < cur_cri_z['x_filter'][2]
    cond2 = y > cur_cri_z['y_filter'][2]
    cond3 = (cur_cri_z['y_coef'] * y) > (cur_cri_z['slope'] * x + cur_cri_z['y_int'])

    # Combine main conditions
    final_mask = cond1 & cond2 & cond3

    # Output
    num_pass = np.sum(final_mask)
    print(f"Number of targets that pass all selection criteria in {cur_cri_z['name']}: {num_pass}")

def distance_original(flux_ori_table, parameters):
    import numpy as np
    import pandas as pd
    from astropy.io import ascii
    from astropy.table import Table
    import os
    # === Prepare original distances from flux_ori_table ===
    detection_band = parameters['detection_band']
    flux_ori_detection = flux_ori_table[f'f_{detection_band}']
    radius_pc = flux_ori_table['Radius_pc']
    spectral_types_ori = flux_ori_table['SpT']
    distance_ori = np.sqrt(flux_ori_detection * radius_pc**2)

    spectral_order = [
        'M7', 'M8', 'M9',
        'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
        'Y0', 'Y1', 'Y2'
        ]

    # === Compute original model quartiles and save ===
    df_ori = pd.DataFrame({'Type': spectral_types_ori, 'Distance': distance_ori})
    available_order = [t for t in spectral_order if t in df_ori['Type'].unique()]
    
    original_summary = []
    for t in available_order:
        vals = df_ori[df_ori['Type'] == t]['Distance']
        if len(vals) > 0:
            low_2, low_1, med, up1, up2 = np.percentile(vals, [2.5, 16, 50, 84, 97.5])
            original_summary.append((t, low_2, low_1, med, up1, up2))
    
    # Save original distance summary table
    output_path_dir = os.path.join(parameters['enzo_path'], "Results", parameters.get('stellar_model'), parameters['fields_name'])
    
    # Ensure output directory exists
    os.makedirs(output_path_dir, exist_ok=True)

    output_path_ori = output_path_dir+'/distance_table_original.tsv'
    ascii.write(
        Table(
            rows=original_summary,
            names=('spectral_type', 'lower_2sigma', 'lower_1sigma', 'median', 'upper_1sigma', 'upper_2sigma')
        ),
        output_path_ori,
        format='tab',
        overwrite=True
    )

    print(f"Saved original distance summary: {output_path_ori}")

def rho_r_disk(r, b, l, Rsun, Zsun, rho0, L_disk, H_disk):
    """
    Compute the stellar density at a given position in the Galaxy based on a double exponential disk model.

    Parameters:
        r (float): Distance from Sun to the object (pc)
        b (float): Galactic latitude (radians)
        l (float): Galactic longitude (radians)
        Rsun (float): Galactocentric distance of the Sun (pc)
        Zsun (float): Vertical height of the Sun above Galactic plane (pc)
        rho0 (float): Local stellar density at the Sun’s position (pc^-3)
        L_disk (float): Radial scale length of the disk (pc)
        H_disk (float): Vertical scale height of the disk (pc)
    Returns:
        float: Density at given (l, b, r), multiplied by r² (for volume element)
    """
    # r is the integration variable (distance in pc)
    X_gc = Rsun-r*np.cos(b)*np.cos(l)
    Y_gc = -r*np.cos(b)*np.sin(l)
    Z_gc = Zsun+r*np.sin(b)
    
    R_gc = np.sqrt(X_gc**2 + Y_gc**2)

    exp_R = np.exp(-(R_gc - Rsun) / L_disk)
    exp_Z = np.exp(-1*np.abs(Z_gc - Zsun) / H_disk)
    rho = rho0 * exp_R * exp_Z
    return rho * r**2

def rho_r_halo(r,b,l, Rsun, Zsun, rho0):
    q = 0.64 #Jurić et al. 2008 See Aganze+2022b
    n = 2.77 #Jurić et al. 2008 See Aganze+2022b
    #calculate galactocentric coordinate
    X_gc = Rsun-r*np.cos(b)*np.cos(l)
    Y_gc = -r*np.cos(b)*np.sin(l)
    Z_gc = Zsun+r*np.sin(b)
    
    R_gc = np.sqrt(X_gc**2 + Y_gc**2)
    denominator = np.sqrt(R_gc**2 + (Z_gc / q)**2)
    rho = rho0 * (Rsun / denominator) ** n

    return rho * r**2

def Area_in_sterradian(value, unit):
    """
    Convert an area specified as (value, unit) to steradians.
    Supported units: 'sr', 'steradian', 'deg2', 'deg^2', 'square_degree',
                     'arcmin2', 'arcsec2'
    """
    unit = str(unit).lower().strip()
    if unit in ('sr', 'steradian', 'steradians'):
        return float(value)
    if unit in ('deg2', 'deg^2', 'square_degree', 'square_degrees', 'deg'):
        # degrees squared -> steradians
        return float(value) * (math.pi / 180.0) ** 2
    if unit in ('arcmin2', 'arcmin^2','arcminsq', 'arcminsquared'):
        # 1 arcmin^2 = (1/60)^2 deg^2
        return float(value) * (1.0 / 60.0) ** 2 * (math.pi / 180.0) ** 2
    if unit in ('arcsec2', 'arcsec^2', 'arcsecsq', 'arcsecsquared'):
        return float(value) * (1.0 / 3600.0) ** 2 * (math.pi / 180.0) ** 2
    raise ValueError(f"Unsupported unit for solid angle: {unit}")

def monte_carlo_aganze(N, b, l,
                        d_min, d_max,
                        Rsun, Zsun,
                        rho0_samples, L_disk,
                        H, H_sigma_lower, H_sigma_upper, type=None,area_sr=1):
    from scipy.integrate import quad

    # Sample all parameters

    if type == 'Thindisk':
        H_disk_samples = np.zeros(N)
        rand_vals = np.random.rand(N)  # values between 0 and 1

        # Negative perturbation (lower half of distribution)
        wneg = np.where(rand_vals < 0.5)[0]
        H_disk_samples[wneg] = H - np.abs(np.random.normal(loc=0, scale=H_sigma_lower, size=len(wneg)))

        # Positive perturbation (upper half of distribution)
        wpos = np.where(rand_vals >= 0.5)[0]
        H_disk_samples[wpos] = H + np.abs(np.random.normal(loc=0, scale=H_sigma_upper, size=len(wpos)))

        # Resample until all H_disk_samples are positive
        while np.any(H_disk_samples <= 0):
            bad_indices = np.where(H_disk_samples <= 0)[0]
            rand_vals = np.random.rand(len(bad_indices))

            wneg = np.where(rand_vals < 0.5)[0]
            wpos = np.where(rand_vals >= 0.5)[0]

            if len(wneg) > 0:
                H_disk_samples[bad_indices[wneg]] = H - np.abs(np.random.normal(0, H_sigma_lower, len(wneg)))
                
            if len(wpos) > 0:
                H_disk_samples[bad_indices[wpos]] = H + np.abs(np.random.normal(0, H_sigma_upper, len(wpos)))

    elif type == 'Thickdisk':
        H_disk_samples = H
    
    # Evaluate the function
    vals = np.zeros(N)
    for i in range(N):
        if type == 'Thindisk':
            vals[i] = quad(rho_r_disk,d_min,d_max,args=(b, l, Rsun, Zsun,
                                rho0_samples[i],
                                L_disk, H_disk_samples[i]))[0]
        elif type == 'Thickdisk':
            vals[i] = quad(rho_r_disk,d_min,d_max,args=(b, l, Rsun, Zsun,
                                rho0_samples[i],
                                L_disk, H_disk_samples))[0]
        elif type == 'Halo':
            vals[i] = quad(rho_r_halo,d_min,d_max, args = (b,l, Rsun, Zsun, rho0_samples[i]))[0]
    
    # Final statistics
    low, median, high = np.percentile(vals, [16, 50, 84]) 
    
    return low, median, high

def number_density(m_detection_arr, parameters, survey_folder):

    """
    Compute number density per spectral type and magnitude bins.

    - m_detection_arr : iterable of magnitudes (centers)
    - parameters : dict expected to contain:
        - 'solid_angle': (value, unit)
        - 'output_path': path
        - 'surveys' : survey subfolder name (used to find distance_table_original.tsv)
        - 'mag_detectionband_bin' : width of magnitude bin
        - 'RA', 'DEC' : target coordinates in degrees
    - survey_folder : base folder used to create 'Analysis/number_density'
    - model : 'Aganze' 
    """
    import matplotlib.cm as cm
    from matplotlib.cm import get_cmap
    import numpy as np
    from astropy.io import ascii
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    N_MC = 100
    analyse_folder = os.path.join(survey_folder, 'Analysis')
    summary_folder = os.path.join(analyse_folder, 'number_density')
    if not os.path.exists(summary_folder):
        os.makedirs(analyse_folder, exist_ok=True)
        os.makedirs(summary_folder, exist_ok=True)

    # Model-dependent constants and tables
    # --- Constants ---
    L_thindisk = 2600.0  # pc (Jurić 2008)
    R0 = 8300.0  # pc
    Z0 = 27.0    # pc

    # load /Users/onnalininnala/Intern/synthesis/1Mj_fit.dat if it exists else fallback
    # while preserving your original idea
    try:
        data_set_path = os.path.join(parameters['enzo_path'], 'data_set')
        rho0_1mj = ascii.read(f'{data_set_path}/1Mj_fit.dat')
        teff_arr = np.array(rho0_1mj['Teff'])
        rho_arr = np.array(rho0_1mj['rho0'])
        sort_idx = np.argsort(teff_arr)
        teff_sorted = teff_arr[sort_idx]
        rho_sorted = rho_arr[sort_idx]
    except Exception:
        # fallback arrays if file missing
        teff_sorted = np.flip(np.array([2025, 1875, 1725, 1575, 1425, 1275, 1125, 975, 825, 675, 525]))
        rho_sorted = np.flip(np.array([0.72, 0.50, 0.78, 0.81, 0.94, 1.95, 1.11, 1.72, 1.99, 2.80, 4.24])) / 1000.

    rho0_T = {
        'temp': np.flip(np.array([2025, 1875, 1725, 1575, 1425, 1275, 1125, 975, 825, 675, 525])),
        'rho0': np.flip(np.array([0.72, 0.50, 0.78, 0.81, 0.94, 1.95, 1.11, 1.72, 1.99, 2.80, 4.24])) / 1000.,
        'rho0_err': np.flip(np.array([0.18, 0.17, 0.20, 0.20, 0.22, 0.30, 0.25, 0.30, 0.32, 0.37, 0.7])) / 1000.
    }

    # Scale heights Table 7 in Aganze+2022b (median values)
    H_thindisk = {'M7': 249, 'M8': 249, 'M9': 249,
                    'L0': 146, 'L1': 146, 'L2': 146, 'L3': 146, 'L4': 146,
                    'L5': 172, 'L6': 172, 'L7': 172, 'L8': 172, 'L9': 172,
                    'T0': 181, 'T1': 181, 'T2': 181, 'T3': 181, 'T4': 181,
                    'T5': 187, 'T6': 187, 'T7': 187, 'T8': 187, 'T9': 187, 'Y0': 187}
    H_plus = {'M7': 48, 'M8': 48, 'M9': 48,
                'L0': 41, 'L1': 41, 'L2': 41, 'L3': 41, 'L4': 41,
                'L5': 175, 'L6': 175, 'L7': 175, 'L8': 175, 'L9': 175,
                'T0': 169, 'T1': 169, 'T2': 169, 'T3': 169, 'T4': 169,
                'T5': 237, 'T6': 237, 'T7': 237, 'T8': 237, 'T9': 237, 'Y0': 237}
    H_neg = {'M7': 61, 'M8': 61, 'M9': 61,
                'L0': 27, 'L1': 27, 'L2': 27, 'L3': 27, 'L4': 27,
                'L5': 56, 'L6': 56, 'L7': 56, 'L8': 56, 'L9': 56,
                'T0': 62, 'T1': 62, 'T2': 62, 'T3': 62, 'T4': 62,
                'T5': 68, 'T6': 68, 'T7': 68, 'T8': 68, 'T9': 68, 'Y0': 68}
    L_thickdisk = 3600.0
    H_thickdisk = 900.0
    rho0_thick_factor = 0.12
    rho0_halo_factor = 0.0025
    spectral_order = [
        'M7', 'M8', 'M9', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'Y0', 'Y1', 'Y2'
    ]

    type_temp = {
        'M7': 2708, 'M8': 2535, 'M9': 2362, 'L0': 2212, 'L1': 2087,
        'L2': 1969, 'L3': 1843, 'L4': 1709, 'L5': 1615, 'L6': 1512,
        'L7': 1434, 'L8': 1324, 'L9': 1269, 'T0': 1261, 'T1': 1238,
        'T2': 1223, 'T3': 1199, 'T4': 1184, 'T5': 1121, 'T6': 971,
        'T7': 838, 'T8': 688, 'T9': 578, 'Y0': 484, 'Y1': 382, 'Y2': 287
    }
    types = list(type_temp.keys())
    temps = list(type_temp.values())
    type_temp_arr = np.array(list(zip(types, temps)), dtype=[('type', 'U3'), ('temp', float)])

    # Area conversion
    solid_ang_value, solid_ang_unit = parameters['solid_angle']
    area_sr = Area_in_sterradian(solid_ang_value, solid_ang_unit)
    print(f"Area_sr={area_sr} steradians")

    full_summary_per_mag = {}
    original_distance_path = os.path.join(parameters['enzo_path'], 'Results', 
                                          parameters['stellar_model'], parameters['fields_name'], 
                                          'distance_table_original.tsv')
    if not os.path.exists(original_distance_path):
        raise FileNotFoundError(f"Distance table not found: {original_distance_path}")
    original_distance_table = ascii.read(original_distance_path, format='tab')
    # ensure spectral_type is string
    original_distance_table['spectral_type'] = original_distance_table['spectral_type'].astype(str)
    
    for t in spectral_order:
        for index_mag, curmag in enumerate(m_detection_arr):
            sel = original_distance_table[original_distance_table['spectral_type'] == t]
            original_med_arr = sel['median']
            original_upper_arr = sel['upper_1sigma']
            original_lower_arr = sel['lower_1sigma']
            if len(original_med_arr) != 1:
                if index_mag == 0:
                    print(f"Warning: No distance row found for spectral type {t}")
                continue

            original_med = float(original_med_arr[0])  # sqrt(flux_original*r^2)
            original_upper = float(original_upper_arr[0])
            original_lower = float(original_lower_arr[0])

            half_mag_bin = parameters['mag_detectionband_bin'] / 2.0
            max_flux_theory = 10.0 ** (-0.4 * ((curmag - half_mag_bin) - 25.0))
            med_flux_theory = 10.0 ** (-0.4 * (curmag - 25.0))
            min_flux_theory = 10.0 ** (-0.4 * ((curmag + half_mag_bin) - 25.0))

            d_min = original_lower / np.sqrt(max_flux_theory)
            d_med = original_med / np.sqrt(med_flux_theory)
            d_max = original_upper / np.sqrt(min_flux_theory)

            # sort descending by temp
            type_temp_arr = np.sort(type_temp_arr, order='temp')[::-1]
            loc_idx = np.where(type_temp_arr['type'] == t)[0]
            if loc_idx.size == 0:
                print(f"Type {t} not found in type_temp_arr; skipping")
                continue
            loc_temp = loc_idx[0]
            curtemp = float(type_temp_arr['temp'][loc_temp])

            # compute rho0 and rho0_err from teff_sorted / rho_sorted if available
            try:
                rho0_values = np.interp(curtemp, teff_sorted, rho_sorted)
            except Exception:
                rho0_values = rho0_T['rho0'][0]  # fallback
            rho0_val = 1.0 * rho0_values / 1000.0  # convert if necessary

            # Interpolate rho0_err from rho0_T table if possible
            rho0_err = 0.0
            if 450 < curtemp < 2100:
                rho0_err = np.interp(curtemp, rho0_T['temp'], rho0_T['rho0_err'])
            elif curtemp >= 2100:
                rho0_err = 0.0
            elif curtemp <= 450:
                rho0_err = 0.0

            # Monte Carlo sampling for rho0
            rho0_samples = np.random.normal(rho0_val, rho0_err, N_MC) if rho0_err > 0 else np.full(N_MC, rho0_val)
            # resample negative if any
            while np.any(rho0_samples <= 0):
                bad = np.where(rho0_samples <= 0)[0]
                rho0_samples[bad] = np.random.normal(rho0_val, rho0_err, size=len(bad)) if rho0_err > 0 else rho0_val

            # get Vertical Scaleheights
            H_thin = H_thindisk.get(t, 187)
            H_sig_plus = H_plus.get(t, 187)
            H_sig_neg = H_neg.get(t, 187)

            # Now integrate with Monte Carlo
            try:
                assert np.isfinite(d_min) and np.isfinite(d_max)
                assert d_min < d_max
                # Convert RA/DEC to Galactic (radians)
                c = SkyCoord(ra=float(parameters['RA']) * u.degree,
                             dec=float(parameters['DEC']) * u.degree, frame='icrs')
                l_rad = c.galactic.l.radian
                b_rad = c.galactic.b.radian

                thin_16, thin_median, thin_84 = monte_carlo_aganze(
                    N_MC, b_rad, l_rad, d_min, d_max,
                    R0, Z0, rho0_samples,  # samples style
                    L_thindisk, H_thin, H_sig_neg, H_sig_plus, type='Thindisk', area_sr=area_sr)
                

                thick_16, thick_median, thick_84 = monte_carlo_aganze(
                    N_MC, b_rad, l_rad, d_min, d_max,
                    R0, Z0, rho0_thick_factor * rho0_samples,
                    L_thickdisk, H_thickdisk, 0, 0, type='Thickdisk', area_sr=area_sr)

                halo_16, halo_median, halo_84 = monte_carlo_aganze(
                    N_MC, b_rad, l_rad, d_min, d_max,
                    R0, Z0, rho0_halo_factor * rho0_samples,
                    0.0, 0.0, 0, 0, type='Halo', area_sr=area_sr)
                    
                # errors and area scaling
                ethin_low = thin_median - thin_16
                ethin_high = thin_84 - thin_median
                ethick_low = thick_median - thick_16
                ethick_high = thick_84 - thick_median
                ehalo_low = halo_median - halo_16
                ehalo_high = halo_84 - halo_median

                # Number density multiplied by survey area
                N_thindisk = area_sr * thin_median
                N_thickdisk = area_sr * thick_median
                N_halo = area_sr * halo_median

                eneg_N_thindisk = area_sr * ethin_low
                epos_N_thindisk = area_sr * ethin_high
                eneg_N_thickdisk = area_sr * ethick_low
                epos_N_thickdisk = area_sr * ethick_high
                eneg_N_halo = area_sr * ehalo_low
                epos_N_halo = area_sr * ehalo_high

                # total
                N_browndwarf_total = N_thindisk + N_thickdisk + N_halo
                err_browndwarf_lower = math.sqrt(eneg_N_thindisk ** 2 + eneg_N_thickdisk ** 2 + eneg_N_halo ** 2)
                err_browndwarf_upper = math.sqrt(epos_N_thindisk ** 2 + epos_N_thickdisk ** 2 + epos_N_halo ** 2)
                
            except Exception as e:
                print(f"Integration failed for {t} at mag={curmag}: {e}")
                N_thindisk = np.nan
                N_thickdisk = np.nan
                N_halo = np.nan
                err_browndwarf_lower = err_browndwarf_upper = np.nan
                N_browndwarf_total = np.nan

                ethin_low = ethin_high = np.nan
                ethick_low = ethick_high = np.nan
                ehalo_low = ehalo_high = np.nan
                thin_median = thick_median = halo_median = np.nan

            # Store result
            full_summary_per_mag.setdefault(curmag, []).append(
                (t, d_min, d_med, d_max, curtemp, 
                ethin_low, thin_median, ethin_high, 
                ethick_low, thick_median, ethick_high,
                ehalo_low, halo_median, ehalo_high,
                eneg_N_thindisk, N_thindisk, epos_N_thindisk,
                eneg_N_thickdisk, N_thickdisk, epos_N_thickdisk,
                eneg_N_halo, N_halo, epos_N_halo,
                err_browndwarf_lower, N_browndwarf_total, err_browndwarf_upper)
            )
            
    # Write final tables
    spectral_data = {}
    for mag, rows in full_summary_per_mag.items():
        table = Table(
            rows=rows,
            names=('spectral_type', 'd_min', 'd_med', 'd_max', 'curtemp', 
                    'eneg_rhor2_thindisk', 'rhor2_thindisk', 'epos_rhor2_thindisk', 
                    'eneg_rhor2_thickdisk', 'rhor2_thickdisk', 'epos_rhor2_thickdisk',
                    'eneg_rhor2_halo', 'rhor2_halo', 'epos_rhor2_halo',
                    'eneg_N_thindisk', 'N_thindisk', 'epos_N_thindisk',
                    'eneg_N_thickdisk', 'N_thickdisk', 'epos_N_thickdisk',
                    'eneg_N_halo', 'N_halo', 'epos_N_halo',
                    'eneg_N_total', 'N_browndwarf_total', 'epos_N_total')
        )
        output_path = os.path.join(summary_folder, f"summary_mag_{mag}.tsv")
        ascii.write(table, output_path, format='tab', overwrite=True)

        for row in table:
            stype = row['spectral_type']
            if stype not in spectral_data:
                spectral_data[stype] = {'mags': [], 'dens': [], 'low': [], 'upper': []}
            spectral_data[stype]['mags'].append(mag)
            spectral_data[stype]['dens'].append(row['N_browndwarf_total'])
            spectral_data[stype]['low'].append(row['eneg_N_total'])
            spectral_data[stype]['upper'].append(row['epos_N_total'])

    # === Plot 1: Number density vs. magnitude (per spectral type) ===
    plt.figure(figsize=(12, 7))
    # --- Use 'tab20' colormap with distinct colors for up to 20 spectral types ---
    spectral_types = list(spectral_data.keys())
    cmap = cm.get_cmap('tab20', len(spectral_types))  # up to 20 unique colors
    color_dict = {stype: cmap(i) for i, stype in enumerate(spectral_types)}

    # --- Plot each spectral type with its assigned color ---
    for stype, data in spectral_data.items():
        mags = np.array(data['mags'])
        dens = np.array(data['dens'])
        low = np.array(data['low'])
        up = np.array(data['upper'])

        if np.allclose(dens, 0):
            continue

        sort_idx = np.argsort(mags)
        mags = mags[sort_idx]
        dens = dens[sort_idx]
        err = np.array([low[sort_idx], up[sort_idx]])

        plt.errorbar(mags, dens, yerr=err, label=stype, fmt='-o',
                    color=color_dict[stype], capsize=3, alpha=0.8)

    plt.xlabel("Magnitude")
    plt.ylabel("Number Density [Number in field]")
    plt.yscale("log")  
    plt.title("Number Density per Spectral Type vs. Magnitude")
    plt.legend(title="Spectral Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{summary_folder}/number_density_mag.png', dpi=300)
    plt.close() 

def analysis(parameters_file, m_detection_arr, mh, survey_folder):

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.gridspec as gridspec

    from astropy.io import ascii
    from astropy.table import Table

    Nmc = 10000

    # =========================================================
    # STYLE
    # =========================================================
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.grid": False,
    })

    # =========================================================
    # PATHS
    # =========================================================
    analyse_folder = os.path.join(survey_folder, "Analysis")
    os.makedirs(analyse_folder, exist_ok=True)

    gal_template = parameters_file["templates"][0]

    # =========================================================
    # STORAGE
    # =========================================================
    eazy_output = []
    eazy_output_err = []

    total_browndwarfs_list = []
    epos_browndwarfs_list = []
    eneg_browndwarfs_list = []

    color_output = {}
    color_output_err = {}

    both_output = {}
    both_output_err = {}

    criteria = None

    output_rows = []

    # =========================================================
    # FUNCTIONS
    # =========================================================
    def compute_pct(eazy, tab, cir):

        mask_color = (tab[cir] == "True")

        mask_both = (
            mask_color &
            (eazy["eazy_best_fit_chi2"] == gal_template)
        )

        N = len(eazy)

        k_color = mask_color.sum()
        k_both = mask_both.sum()

        p_color = k_color / N
        p_both = k_both / N

        err_color = np.sqrt(
            p_color * (1 - p_color) / N
        )

        err_both = np.sqrt(
            p_both * (1 - p_both) / N
        )

        return (
            p_color * 100,
            p_both * 100,
            err_color * 100,
            err_both * 100,
        )

    def beautify_axis(ax):

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(direction="in")


    # =========================================================
    # MAIN LOOP
    # =========================================================
    for m in m_detection_arr:

        folder_base = os.path.join(
            survey_folder,
            f"mag_{m}"
        )

        # =====================================================
        # EAzY
        # =====================================================
        eazy_file = (
            f"{folder_base}/Eazy/"
            f"{mh}_bestfit_{m}.dat"
        )

        eazy = ascii.read(eazy_file)

        chi2_dict = dict(
            zip(
                *np.unique(
                    eazy["eazy_best_fit_chi2"],
                    return_counts=True
                )
            )
        )

        k_eazy = chi2_dict.get(gal_template, 0)

        N_eazy = len(eazy)

        p_eazy = k_eazy / N_eazy

        pct_gal = p_eazy * 100

        err_eazy = np.sqrt(
            p_eazy * (1 - p_eazy) / N_eazy
        ) * 100

        eazy_output.append(pct_gal)
        eazy_output_err.append(err_eazy)

        # =====================================================
        # Number Density
        # =====================================================
        df = pd.read_csv(
            f"{analyse_folder}/number_density/"
            f"summary_mag_{m}.tsv",
            sep="\t"
        )

        td = df["N_browndwarf_total"].sum()
        td_neg = df["eneg_N_total"].sum()
        td_pos = df["epos_N_total"].sum()

        total_browndwarfs_list.append(td)
        eneg_browndwarfs_list.append(td_neg)
        epos_browndwarfs_list.append(td_pos)

        # =====================================================
        # Color Criteria
        # =====================================================
        color_folder = f"{folder_base}/color"

        files = (
            f"{mh}_{parameters_file['color_criteria']}.dat"
        )

        tab = ascii.read(os.path.join(color_folder, files))

        if len(tab.colnames) <= 1:
            continue

        cirs = tab.colnames[1:]

        criteria = cirs

        # =====================================================
        # TABLE ROW
        # =====================================================
        row = {
            "mag": m,
            "Number_density": td,
            "N_err_lt": td_neg,
            "N_err_gt": td_pos,
            "fraction_eazy": pct_gal,
            "err_eazy": err_eazy,
        }

        for cir in cirs:

            color_pct, both_pct, color_err, both_err = (
                compute_pct(eazy, tab, cir)
            )

            color_output.setdefault(cir, []).append(color_pct)
            color_output_err.setdefault(cir, []).append(color_err)

            both_output.setdefault(cir, []).append(both_pct)
            both_output_err.setdefault(cir, []).append(both_err)

            # ----------------------------------------------
            # Save dynamically into row
            # ----------------------------------------------
            row[f"color_{cir}"] = color_pct
            row[f"err_color_{cir}"] = color_err

            row[f"both_{cir}"] = both_pct
            row[f"err_both_{cir}"] = both_err

        output_rows.append(row)

    # =========================================================
    # SAVE TABLE
    # =========================================================
    output_tab = Table(rows=output_rows)

    ascii.write(
        output_tab,
        f"{analyse_folder}/"
        f"result_table_{parameters_file['fields_name']}_{mh}.dat",
        format="tab",
        overwrite=True
    )

    # =========================================================
    # ARRAYS
    # =========================================================
    td = np.array(total_browndwarfs_list)

    td_pos = np.array(epos_browndwarfs_list)

    td_neg = np.array(eneg_browndwarfs_list)

    err_td = np.array([td_neg, td_pos])

    # =========================================================
    # COLORS
    # =========================================================
    colors = [
        cm.tab10(i / max(len(criteria)-1, 1))
        for i in range(len(criteria))
    ]

    # =========================================================
    # FIGURE
    # =========================================================
    fig = plt.figure(
        figsize=(15, 5),
        constrained_layout=True
    )

    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.2, 1.2, 1.2, 1.5]
    )

    axs = [
        fig.add_subplot(gs[0, i])
        for i in range(4)
    ]

    # =========================================================
    # PANEL 1
    # =========================================================
    ax = axs[0]

    ax.errorbar(
        m_detection_arr,
        td,
        yerr=err_td,
        fmt="o-",
        color="darkviolet",
        capsize=4,
    )

    ax.set_xlabel("Detection Magnitude")
    ax.set_ylabel(r"$N\_brown\ dwarf$")

    beautify_axis(ax)

    # =========================================================
    # PANEL 2
    # =========================================================
    ax = axs[1]

    ax.errorbar(
        m_detection_arr,
        eazy_output,
        yerr=eazy_output_err,
        fmt="o-",
        color="deepskyblue",
        capsize=4,
    )

    ax.set_xlabel("Detection Magnitude")
    ax.set_ylabel("Fraction (%)")

    beautify_axis(ax)

    # =========================================================
    # PANEL 3
    # =========================================================
    ax = axs[2]

    for i, cir in enumerate(criteria):

        ax.errorbar(
            m_detection_arr,
            color_output[cir],
            yerr=color_output_err[cir],
            fmt="o-",
            color=colors[i],
            capsize=4,
            label=cir,
        )

    ax.set_xlabel("Detection Magnitude")
    ax.set_ylabel("Pass Fraction (%)")

    beautify_axis(ax)

    # =========================================================
    # PANEL 4
    # =========================================================
    ax = axs[3]

    summary_lines = []

    for i, cir in enumerate(criteria):

        frac = np.array(both_output[cir]) / 100.0
        frac_err = np.array(both_output_err[cir]) / 100.0

        y = []

        yerr_low = []
        yerr_high = []

        for j in range(len(td)):

            # -------------------------------------------------
            # Sample number density
            # -------------------------------------------------
            td_samples = np.random.normal(
                td[j],
                0.5 * (td_pos[j] + td_neg[j]),
                Nmc
            )

            td_samples = np.clip(td_samples, 0, None)

            # -------------------------------------------------
            # Sample contamination fraction
            # -------------------------------------------------
            frac_samples = np.random.normal(
                frac[j],
                frac_err[j],
                Nmc
            )

            frac_samples = np.clip(
                frac_samples,
                0,
                1
            )

            # -------------------------------------------------
            # Combined contamination
            # -------------------------------------------------
            contam_samples = td_samples * frac_samples

            p16, p50, p84 = np.percentile(
                contam_samples,
                [16, 50, 84]
            )

            y.append(p50)

            yerr_low.append(p50 - p16)
            yerr_high.append(p84 - p50)

        y = np.array(y)

        err_y = np.array([
            yerr_low,
            yerr_high
        ])

        # -----------------------------------------------------
        # Plot
        # -----------------------------------------------------
        ax.errorbar(
            m_detection_arr,
            y,
            yerr=err_y,
            fmt="o-",
            color=colors[i],
            capsize=4,
            label=cir,
        )

        # -----------------------------------------------------
        # Total contamination summary
        # -----------------------------------------------------
        total_contam = np.sum(y)

        total_err = np.sqrt(
            np.sum(
                (
                    0.5 * (
                        np.array(yerr_low) +
                        np.array(yerr_high)
                    )
                )**2
            )
        )

        summary_text = (
            f"{cir} "
            f"Σ={total_contam:.4e} ± {total_err:.3e}"
        )

        print(summary_text)

        summary_lines.append(summary_text)

    ax.set_xlabel("Detection Magnitude")

    ax.set_ylabel(r"$N\_contamination$")

    ax.legend(
        title="Criteria",
        fontsize=11,
        frameon=True,
    )

    beautify_axis(ax)

    # =========================================================
    # SAVE FIGURE
    # =========================================================
    output_name = (
        f"{analyse_folder}/"
        f"{parameters_file['fields_name']}"
        f"_contaminate_{mh}.png"
    )

    fig.savefig(
        output_name,
        dpi=500,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)
    # =========================================================
    # SAVE SUMMARY TEXT
    # =========================================================
    summary_file = (
        f"{analyse_folder}/"
        f"{parameters_file['fields_name']}"
        f"_contamination_summary_{mh}.txt"
    )

    with open(summary_file, "w") as f:

        f.write("# Contamination Summary\n")
        f.write(f"# Field: {parameters_file['fields_name']}\n")
        f.write(f"# Spectral Type: {mh}\n\n")

        for line in summary_lines:
            f.write(line + "\n")
    
    # =========================================================
    # RETURN RESULT TO GUI
    # =========================================================
    summary_message = "\n".join(summary_lines)

    return output_name, summary_message
    
    