# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:08:23 2024

@author: daniela
"""


#depth of observation - max value of magnitude, maybe adjustable (IN PROGRESS)
#script prints max magnitude -> decide how to filter by hand (DONE)
#figure out errors
#errors on the profile, error in bin N/A +/- rootN/ (SQRT DONE)
#A
#jwst vs hst etc
# check if there is a specific flag that shows which CCD the observation is from!

#perform necessary imports
import pandas as pd
from physt import polar
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import linregress
 

class GALAXY:
    def __init__(self, name, morph, obs, df, RA, DEC, dist, f1_name, f2_name, 
                 f1_flag, f2_flag, pw_radius):
        """
        Initialize a Galaxy class in order to make Galaxy objects

        Parameters
        ----------
        name : str
            Galaxy name.
        morph : str
            galactic morphology.
        obs : str
            observation type ie HST/HLA or JWST.
        df : DataFrame
            data frame of downloaded data from MAST.
        RA : float
            RA of the galaxy center from MAST.
        DEC : float
            DEC of the galaxy center from MAST.
        dist : float
            distance to the galaxy from the wiki page(?).
        f1_name : str
            name of the lower magnitude filter.
        f2_name : str
            name of the upper magnitude filter.
        f1_flag : str
            name of the lower filter flag.
        f2_flag : str
            name of the upper filter flag.
        pw_radius : float
            radius dividing piecewise distribution of radial distance

        Returns
        -------
        None.

        """

        self.name = name
        self.morph = morph
        self.obs = obs
        self.df = df
        self.RA = RA
        self.DEC = DEC
        self.dist = dist
        self.f1_name = f1_name
        self.f2_name = f2_name
        self.f1_flag = f1_flag
        self.f2_flag = f2_flag
        self.pw_radius = pw_radius
        
        
    
    # Instance method
    def description(self):
        return f"{self.name} is a {self.obs} observation."


class ISOCHRONE:
    def __init__(self, age, df, dist, f1_name, f2_name):
        """
        Initialize an Isochrone class in order to generate theory tracks from the BaSTci model

        Parameters
        ----------
        age : float
            Isochrone age, used as name.
        df : DataFrame
            data frame of downloaded data from MAST.
        dist : float
            distance to the galaxy from the wiki page(?).
        f1_name : str
            name of the lower magnitude filter.
        f2_name : str
            name of the upper magnitude filter.

        Returns
        -------
        None.

        """

        self.age = age
        self.df = df
        self.dist = dist
        self.f1_name = f1_name
        self.f2_name = f2_name
        
    # Instance method
    def description(self):
        return f"This isochrone is for a system {self.name}Myr old."



# Function to remove rows with any NaN or infinite values
def clean_df(df):
    """Remove rows containing any NaN or infinite values"""
    # Convert infinities to NaN first
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop rows with any NaN values
    cleaned_df = df.dropna()
    return cleaned_df

def find_trgb(dist):
    #use paper to convert from trgb in I/V/B to filter
    #call distance modulus to find app mag
    abs_mag=0
    app_mag(abs_mag, dist)
    return


def find_lim_mag(magnitudes, trgb_mag):
    """
    Cumulative sum approach, TRGB handling
    
    Parameters:
    magnitudes : array of magnitudes
    trgb_mag : TRGB magnitude cutoff
    """
    #initialize decay rate and min points in the fit to handle error cases
    decay_rate=1/2 #divergence threshold (0.5 = 1/2)
    min_points=10 #minimum points required for analysis
    
    # Sort magnitudes and create cumulative sums
    sorted_mags = np.sort(magnitudes)
    cumul_sums = np.cumsum(sorted_mags)
    norm_sums = cumul_sums / cumul_sums[-1]  # Normalized to 1
    
    # Find TRGB index and select data dimmer than TRGB
    trgb_idx = np.searchsorted(sorted_mags, trgb_mag)
    mags_dimmer_trgb = sorted_mags[trgb_idx:]
    sums_dimmer = norm_sums[trgb_idx:]
    
    if len(mags_dimmer_trgb) < min_points:
        raise ValueError(f"Only {len(mags_dimmer_trgb)} points below TRGB (needs {min_points})")
    
    # Find where cumulative sum drops below (1 - decay_rate) of maximum
    threshold = 1 - decay_rate
    for i in range(len(sums_dimmer)-1, -1, -1):
        if sums_dimmer[i] <= threshold:
            limiting_mag = mags_dimmer_trgb[i]
            break
    else:
        limiting_mag = mags_dimmer_trgb[-1]
    
    return limiting_mag, sorted_mags, norm_sums



#create filters to remove non-target objects
#constrain the displayed data by constraining the magnitude
#make more specific filter for each data set


def filter_stars(df, Galaxy):
    """
    Filter by catalogue flag. We want only the objects wth a "0" flag (stars).

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    Galaxy : Galaxy
        DESCRIPTION.

    Returns
    -------
    df1 : dataframe
        DESCRIPTION.

    """
    id_notAStar = (df[Galaxy.f1_flag]!=0)|(df[Galaxy.f2_flag]!=0)
    df1 = df[id_notAStar]
    return df1

def filter_mag(df, Galaxy, trgb, limit_f1, limit_f2):
    """
    Filter by magnitudes: between tip of red giant branch and limiting magnitude.

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    Galaxy : Galaxy
        DESCRIPTION.

    Returns
    -------
    df1 : dataframe
        DESCRIPTION.

    """
    #y-axis is the larger filter
    filter1 = ((df[Galaxy.f1_name] > trgb) & (df[Galaxy.f1_name] < limit_f1))
    filter2 = ((df[Galaxy.f2_name] > trgb) & (df[Galaxy.f2_name] < limit_f2))
    #will have to write in terms of distance
    #tip of red giant branch fixed, changes based on apparent mag
    df1 = df[filter1]
    df1 = df1[filter2]
    return df1


def filter_rad(radius, r, df, Galaxy):
    """
    After calculating the radius for each point, consider piecewise separation to better take limiting magnitudes

    Parameters
    ----------
    radius : array
        DESCRIPTION.
    r : float
        DESCRIPTION.
    df : dataframe
        DESCRIPTION.
    Galaxy : Galaxy
        DESCRIPTION.

    Returns
    -------
    df_in : dataframe
        DESCRIPTION.
    df_out : dataframe
        DESCRIPTION.

    """
    filter_in = (radius <= r)
    filter_out = (radius > r)
    df_in = df[filter_in]
    df_out = df[filter_out]
    return df_in, df_out



def cart2pol(x, y):
    """
    Converts cartesian coordinates (x, y) to polar coordinates (rho, theta)

    Parameters
    ----------
    x : float array
        x coordinate of location
    y : float array
        y coordinate of location

    Returns
    -------
    rho : float array
        radius from origin
    theta : float array
        angle counterclockwise from x-axis

    """
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return rho, theta

def conv_loc(Galaxy):
    """
    Convert RA, DEC units

    Parameters
    ----------
    Galaxy : Galaxy
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # in min(hrs) from min(deg)
    Galaxy.df.RA = (Galaxy.df.RA - Galaxy.RA)*24.0
    # in arcmin (from deg)
    Galaxy.df.DEC = (Galaxy.df.DEC - Galaxy.DEC)*60.0
    return 

def app_mag(age, filter_1, filter_2, distance):
    """
    Calculate apparent magnitude for theoretical data using distance modulus

    Parameters
    ----------
    filt : float array
        absolute magnitude in a particular filter
    distance : float
        distance of galaxy from Earth

    Returns
    -------
    app : float array
        apparent magnitude in a particular filter

    """
    mag_f1 = filter_1 + 5*np.log10(distance/10)
    mag_f2 = filter_2 + 5*np.log10(distance/10)
    
    return mag_f1, mag_f2

def process_glx(g):
    """
    Convert units, clean the data frames, then apply filters and get radial density distributions
    
    Parameters
    ----------
    g : Galaxy
        a specific galaxy object.

    Returns
    -------
    None.

    """
    #CONVERT RA, DEC units
    conv_loc(g)


    #CLEAN NaN VALUES
    g.fdf = clean_df(g.df)
    
    
    #TRGB: global value for abs mag of TRGB for nearby dwarf galaxies: -4 +-0.1 in I band
    TRGB = -4 + 5*np.log10(g.dist/10)
    print(TRGB)
    
    
    #LIMITING MAGNITUDE
    lim_mag_f1, sort_f1, norm_sum_f1 = find_lim_mag(g.fdf[g.f1_name], TRGB)
    lim_mag_f2, sort_f2, norm_sum_f2 = find_lim_mag(g.fdf[g.f2_name], TRGB)


    #APPLY DATA FILTERS 
    obs_df1 = filter_stars(g.fdf, g)
    #obs_df2 = filter_color(obs_df1, g)
    obs_df3 = filter_mag(obs_df1, g, TRGB, lim_mag_f1, lim_mag_f2)
    g.fdf = obs_df3
    

    #APPARENT MAGNITUDE: THEORETCIAL DATA
    #calculate the apparent magnitude for both wavelength filters
    app_A_f1, app_A_f2 = app_mag(12000, df_12000.F555W, df_12000.F814W, g.dist)
    app_B_f1, app_B_f2 = app_mag(2000, df_2000.F555W, df_2000.F814W, g.dist)
    

    #COLOR INDICES
    #smaller minus larger wavelength filter
    #unfiltered
    col_idx = g.df[g.f1_name] - g.df[g.f2_name]
    #filtered
    f_col_idx = g.fdf[g.f1_name] - g.fdf[g.f2_name]
    #theory
    idx_A = app_A_f1 - app_A_f2
    idx_B = app_B_f1 - app_B_f2


    
    # CUMULATIVE DISTRIBUTION of MAGNITUDES 
    fig_cu_mag, (ax_cu_mag1, ax_cu_mag2) = plt.subplots(1, 2, figsize=(12, 6))
    fig_cu_mag.suptitle(f"{g.name} Cumulative Distribution of Magnitudes")
    
    # Filter 1
    ax_cu_mag1.scatter(sort_f1, norm_sum_f1, color='black', s=0.5, label='data')
    ax_cu_mag1.axvline(lim_mag_f1, color='red', linestyle='--', label=f'Limiting Magnitude: {lim_mag_f1:.2f}')
    ax_cu_mag1.axvline(TRGB, color='purple', linestyle=':',label=f'TRGB: {TRGB:.2f}')
    ax_cu_mag1.set_title(g.f1_name)
    ax_cu_mag1.set_xlabel("Magnitude")
    ax_cu_mag1.set_ylabel("Cumulative Magnitude")
    ax_cu_mag1.set_yscale('log')
    ax_cu_mag1.legend()
    
    # Filter 2
    ax_cu_mag2.scatter(sort_f2, norm_sum_f2, color='black', s=0.5, label='data')
    ax_cu_mag2.axvline(lim_mag_f2, color='red', linestyle='--', label=f'Limiting Magnitude: {lim_mag_f2:.2f}')
    ax_cu_mag2.axvline(TRGB, color='purple', linestyle=':',label=f'TRGB: {TRGB:.2f}')
    ax_cu_mag2.set_title(g.f2_name)
    ax_cu_mag2.set_xlabel("Magnitude")
    ax_cu_mag2.set_ylabel("Cumulative Magnitude")
    ax_cu_mag2.set_yscale('log')
    ax_cu_mag2.legend()
    
    plt.show()



    
    #HR DIAGRAMS
    fig_HR, (ax_HR1, ax_HR2) = plt.subplots(1, 2, figsize=(12, 6))
    fig_HR.suptitle(f"{g.name} HR diagram: Observed and theoretical data")
    
    # Unfiltered
    ax_HR1.scatter(col_idx, g.df[g.f2_name], color='black', s=0.5, label='observed')
    ax_HR1.scatter(f_col_idx, g.fdf[g.f2_name], color='r', s=0.5, label='observed: filtered')
    ax_HR1.scatter(idx_A, app_A_f2, color='b', s=0.5, label='theory: 12000Myr')
    ax_HR1.scatter(idx_B, app_B_f2, color='g', s=0.5, label='theory: 2000Myr')
    ax_HR1.invert_yaxis()
    ax_HR1.set_ylim(28, 20)
    ax_HR1.set_title("Unfiltered")
    ax_HR1.set_xlabel(f"Color index ({g.f1_name} - {g.f2_name})") #get name from Galaxy objext
    ax_HR1.set_ylabel("Magnitude")
    ax_HR1.legend()

    # Filtered
    ax_HR2.scatter(f_col_idx, g.fdf[g.f2_name], color='r', s=0.5, label='observed')
    ax_HR2.scatter(idx_A, app_A_f2, color='b', s=0.5, label='theory: 12000Myr')
    ax_HR2.scatter(idx_B, app_B_f2, color='g', s=0.5, label='theory: 2000Myr')
    ax_HR2.invert_yaxis()
    ax_HR2.set_ylim(28, 20)
    ax_HR2.set_title("Filtered")
    ax_HR2.set_xlabel(f"Color index ({g.f1_name} - {g.f2_name})")
    ax_HR2.set_ylabel("Magnitude")
    ax_HR2.legend()

    plt.tight_layout()
    plt.show()




    #LOCATION
    #RA-DEC location diagram of the entire galaxy
    fig_loc, (ax_loc1, ax_loc2) = plt.subplots(1, 2, figsize=(12, 6))
    fig_loc.suptitle(f"{g.name} Location plot: RA vs DEC")

    #unfiltered
    ax_loc1.scatter(g.df.RA, g.df.DEC, color='b', s=0.1, label = 'observed (only in unfiltered)')
    ax_loc1.scatter(g.fdf.RA, g.fdf.DEC, color='red', s=0.1, label='observed')
    ax_loc1.set_xlabel("RA (minutes)")
    ax_loc1.set_ylabel("DEC (degrees)")
    ax_loc1.legend()

    #filtered
    ax_loc2.scatter(g.fdf.RA, g.fdf.DEC, color='black', s=0.1, label = 'observed')
    ax_loc2.set_xlabel("RA (minutes)")
    ax_loc2.set_ylabel("DEC (degrees)")
    ax_loc2.legend()

    plt.tight_layout()
    plt.legend()




    #POLAR DENSITY PLOT
    # Define binning
    n_theta_bins = 30
    n_radial_bins = 20
    r_start = 0
    r_end = 2.3 
    a_start = -np.pi 
    a_end = np.pi

    rbins = np.linspace(r_start, r_end, n_radial_bins)
    abins = np.linspace(a_start, a_end, n_theta_bins)
    A, R = np.meshgrid(abins, rbins)

    fig_pol, (ax_pol1, ax_pol2) = plt.subplots(1, 2, subplot_kw=dict(projection="polar"), figsize=(12, 6))
    fig_pol.suptitle(f"{g.name} Polar Density Plot")

    #unfiltered
    radius1, angle1 = cart2pol(g.df.RA, g.df.DEC)
    hist1, hist1x, hist1y = np.histogram2d(angle1, radius1, bins=(abins, rbins), weights=1/radius1)
    pc1 = ax_pol1.pcolormesh(A, R, hist1.T, cmap="magma_r")
    fig_pol.colorbar(pc1, ax=ax_pol1)
    ax_pol1.grid(True)
    ax_pol1.set_title("Unfiltered")
    
    #filtered
    radius2, angle2 = cart2pol(g.fdf.RA, g.fdf.DEC)
    hist2, hist2x, hist2y = np.histogram2d(angle2, radius2, bins=(abins, rbins), weights=1/radius2)
    pc2 = ax_pol2.pcolormesh(A, R, hist2.T, cmap="magma_r")
    fig_pol.colorbar(pc2, ax=ax_pol2)
    ax_pol2.grid(True)
    ax_pol2.set_title("Filtered")
    plt.show()
    
    """
    #find limiting magnitude based on piecewise separation
    #currently works for unfiltered, BUT not for filtered data
    m_in, m_out = filter_rad(radius2, 0.6, g.fdf, g)
    print('inner section, 814: ',max(m_in[g.f2_name]))
    print('outer section 814: ',max(m_out[g.f2_name]))
    print('inner, 555: ',max(m_in[g.f1_name]))
    print('outer 555: ',max(m_out[g.f1_name]))
    """



    #RADIAL DENSITY PROFILE
    #unfiltered
    fig_prof, (ax_prof1, ax_prof2) = plt.subplots(1, 2, figsize=(12, 6))
    fig_prof.suptitle(f"{g.name} Radial Density Plot (density as a function of radius without considering 'zero' bins)")
    
    rbins_c = (rbins[1:]+rbins[:-1])/2
    
    # Create a mask for r <= piecewise radius and r > piecewise radius
    mask_r = rbins_c <= g.pw_radius
    
    profile = rbins_c*0.0
    #for i in range(n_theta_bins-1):
        #ax_prof1.plot(rbins_c, hist1[i,:])
    for i in range(n_radial_bins-1):
        count = 0
        for j in range(n_theta_bins-1):
            #if histogram value is nonzero...
            if hist1[j,i] != 0:
                profile[i] += hist1[j,i]
                count += 1
        #density as a function of radius without considering "zero" bins
        profile[i] /= count
    err_prof1 = np.sqrt(profile)  # Poisson noise error is sqrt of the count
    # Split the profiles at r = 0.6 and plot each part separately
    ax_prof1.errorbar(rbins_c[mask_r], profile[mask_r], yerr=err_prof1[mask_r], fmt='-', color='orange', lw=3, label=f'(r <= {g.pw_radius})')
    ax_prof1.errorbar(rbins_c[~mask_r], profile[~mask_r],yerr=err_prof1[~mask_r], fmt='-', color='blue', lw=3, label=f'(r > {g.pw_radius})')
    plt.yscale('log')
    ax_prof1.set_title("UNFILTERED")
    ax_prof1.set_xlabel("Radial distance")
    ax_prof1.set_ylabel("Counts")
    ax_prof1.legend()

    #filtered
    profile2 = rbins_c*0.0
    #for i in range(n_theta_bins-1):
        #ax_prof2.plot(rbins_c, hist2[i,:])
    for i in range(n_radial_bins-1):
        count = 0
        for j in range(n_theta_bins-1):
            #if histogram value is nonzero...
            if hist1[j,i] != 0:
                profile2[i] += hist2[j,i]
                count += 1
        #density as a function of radius without considering "zero" bins
        profile2[i] /= count
    err_prof2 = np.sqrt(profile2)
    ax_prof2.errorbar(rbins_c[mask_r], profile2[mask_r], yerr=err_prof2[mask_r], fmt='-', color='orange', lw=3, label=f'(r <= {g.pw_radius})')
    ax_prof2.errorbar(rbins_c[~mask_r], profile2[~mask_r], yerr=err_prof2[~mask_r], fmt='-', color='blue', lw=3, label=f'(r > {g.pw_radius})')
    plt.yscale('log')
    ax_prof2.set_title("FILTERED")
    ax_prof2.set_xlabel("Radial distance")
    ax_prof2.set_ylabel("Counts")
    ax_prof2.legend()

    plt.tight_layout()
    plt.show()
    
    fig = plt.figure()
    plt.title(f"{g.name} Radial density plot")
    plt.ylabel("Counts")
    plt.xlabel("Radius")
    plt.errorbar(rbins_c[mask_r], profile[mask_r], yerr=err_prof1[mask_r], fmt='-', color='orange', lw=3, label=f'unfiltered(r <= {g.pw_radius})')
    plt.errorbar(rbins_c[~mask_r], profile[~mask_r],yerr=err_prof1[~mask_r], fmt='-', color='blue', lw=3, label=f'(r > {g.pw_radius})')
    plt.errorbar(rbins_c[mask_r], profile2[mask_r], yerr=err_prof2[mask_r], fmt='-', color='red', lw=3, label=f'filtered(r <= {g.pw_radius})')
    plt.errorbar(rbins_c[~mask_r], profile2[~mask_r], yerr=err_prof2[~mask_r], fmt='-', color='green', lw=3, label=f'(r > {g.pw_radius})')
    plt.yscale('log')
    plt.legend()
    plt.show() 
    
    return(f"{g.name} processed")
  

    


"""RUN THE WHOLE THING"""
#simplified file path
path = "C:/Users/daniela/Documents/Ricotti Research/"

#LOAD THEORETICAL DATA, (BASTci) at 12000Myr, 2000Myr age
df_12000 = pd.read_csv(path + "Theory Isochrones/12000Myr.isc_wfpc2", sep='\s+')
df_2000 = pd.read_csv(path + "Theory Isochrones/2000Myr.isc_wfpc2", sep='\s+')

#INITIALIZE ISOCHRONE OBJECTS
iso12000 = ISOCHRONE(12000, df_12000, 2.69e6, "F555W", "F814W")


#LOAD OBSERVATIONAL DATA
df_ugca133_W = pd.read_csv(path + "Data/UCGA133/UGCA133_wfpc2_wfc.csv")
df_ugca133 = pd.read_csv(path + "Data/UCGA133/UGCA133_acs_wfc.csv")
df_ugc9128 = pd.read_csv(path + "Data/UGC9128/UGC9128_acs_wfc.csv")
df_ugc8508 = pd.read_csv(path + "Data/UGC8508/UGC8508_acs_wfc.csv")

#INITIALIZE GALAXY OBJECTS
ugca133_W = GALAXY("UGCA133", "Im", "WFPC2/WFC", df_ugca133_W, 113.5479167, 
                 66.8797222, 2.96e6, "f555w_TOTMAG", "f814w_TOTMAG", 
                 "f555w_FLAGS", "f814w_FLAGS", 0.6)

ugca133 = GALAXY("UGCA133", "Im", "ACS/WFC", df_ugca133, 113.5479167, 
                 66.8797222, 2.96e6, "f475w_MAGAUTO","f814w_MAGAUTO", 
                 "f475w_FLAGS", "f814w_FLAGS", 0.6)

ugc9128 = GALAXY("UGC9128", "ImIV-V", "ACS/WFC", df_ugc9128, 213.987847, 
                 23.058297, 2.24e6, "f606w_MAGAUTO", "f814w_MAGAUTO", 
                 "f606w_FLAGS", "f814w_FLAGS", 0.6)

ugc8508 = GALAXY("UGC8508", "IAm", "ACS/WFC", df_ugc8508, 202.686951, 
                 54.910701, 2.69e6, "f475w_MAGAUTO","f814w_MAGAUTO", 
                 "f475w_FLAGS", "f814w_FLAGS", 0.6)

#ARRAY OF GALAXY OBJECTS
glx = [ugca133, ugc9128, ugc8508]

#PROCESS DATA
process_glx(ugc8508)