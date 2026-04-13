# -*- coding: utf-8 -*-
"""
JWST Galaxy Analysis Pipeline
Author: Daniela Sofia Colombi
University of Maryland - College Park
April 13, 2026

Purpose: Process JWST NIRCAM observations to pull out the stellar halos from
         nearby dwarf galaxies using extendedness, concentration index,
         ellipticity, and theoretical isochrone matching.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import glob
import os
from scipy.interpolate import interp1d


# ============================================================================
# CONSTANTS & GLOBAL SETTINGS
# ============================================================================
# JWST NIRCAM filters (both short and long wavelength channels)
JWST_filters = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
                'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W',
                'F460M', 'F466N', 'F470N', 'F480M']


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

class GALAXY:
    """Container for galaxy properties from MAST catalog"""
    def __init__(self, name, RA, DEC, dist):
        self.name = name      # Galaxy name (used for folder access)
        self.RA = RA          # Right Ascension (degrees)
        self.DEC = DEC        # Declination (degrees)
        self.dist = dist      # Distance to galaxy (Mpc)


class ISOCHRONE:
    """Container for theoretical isochrone data from BaSTI models"""
    def __init__(self, age, df, dist):
        self.age = age        # Isochrone age (Myr)
        self.df = df          # Dataframe with theoretical magnitudes
        self.dist = dist      # Distance for apparent magnitude calculation


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def app_mag(age, filter_1, filter_2, distance):
    """
    Convert absolute magnitudes to apparent magnitudes using distance modulus.
    
    Parameters:
    -----------
    filter_1, filter_2 : array-like
        Absolute magnitudes in each filter
    distance : float
        Distance to galaxy (Mpc)
    
    Returns:
    --------
    mag_f1, mag_f2 : array-like
        Apparent magnitudes
    """
    mag_f1 = filter_1 + 5 * np.log10(distance / 10)
    mag_f2 = filter_2 + 5 * np.log10(distance / 10)
    return mag_f1, mag_f2


def read_jwst_ecsv(filename):
    """Read JWST ECSV catalog file and return as pandas DataFrame"""
    try:
        table = Table.read(filename, format='ascii.ecsv')
        df = table.to_pandas()
        print(f"Successfully loaded {filename}")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def load_all_filters():
    """
    Load catalogs for two specified filters from current directory.
    Uses global variables F1 and F2 for filter names.
    """
    target_filters = [F1, F2]
    filter_files = glob.glob('*cat.ecsv')
    filter_data = {}
    
    for file in filter_files:
        for target in target_filters:
            if target in file.upper():
                print(f"Loading {target}...")
                df = read_jwst_ecsv(file)
                if df is not None:
                    filter_data[target] = df
                break
        if len(filter_data) == len(target_filters):
            break
            
    return filter_data


def match_sources_across_filters(filter_data):
    """
    Match sources between two filters based on RA/DEC coordinates.
    Uses global variables: F1, F2, match_radius_arcsec.
    
    Returns:
    --------
    matched_df : DataFrame with combined photometry and morphology flags
    """
    if F1 not in filter_data or F2 not in filter_data:
        print(f"Error: {F1} or {F2} not found in filter data")
        return None
    
    df1 = filter_data[F1].copy()
    df2 = filter_data[F2].copy()
    
    # Convert arcseconds to degrees for matching
    match_radius_deg = match_radius_arcsec / 3600.0
    
    print(f"Matching sources between {F1} and {F2}")
    print(f"{F1}: {len(df1)} sources, {F2}: {len(df2)} sources")
    
    matched_sources = []
    
    for idx1, row1 in df1.iterrows():
        ra1, dec1 = row1['sky_centroid.ra'], row1['sky_centroid.dec']
        
        # Find closest match in second filter
        distances = np.sqrt((df2['sky_centroid.ra'] - ra1)**2 + 
                           (df2['sky_centroid.dec'] - dec1)**2)
        min_idx = distances.idxmin()
        min_distance = distances[min_idx]
        
        if min_distance <= match_radius_deg:
            row2 = df2.loc[min_idx]
            matched_row = {
                'RA': ra1,
                'DEC': dec1,
                'separation_arcsec': min_distance * 3600.0,
                f'{F1}_mag': row1.get('aper50_vegamag', np.nan),
                f'{F2}_mag': row2.get('aper50_vegamag', np.nan),
                f'{F1}_is_extended': row1.get('is_extended', np.nan),
                f'{F2}_is_extended': row2.get('is_extended', np.nan),
                f'{F1}_CI': row1.get('CI_50_30', np.nan),
                f'{F2}_CI': row2.get('CI_50_30', np.nan),
                f'{F1}_ellipticity': row1.get('ellipticity', np.nan),
                f'{F2}_ellipticity': row2.get('ellipticity', np.nan)
            }
            matched_sources.append(matched_row)
    
    matched_df = pd.DataFrame(matched_sources)
    
    if len(matched_df) > 0:
        # Calculate color index (F1 - F2)
        matched_df['color_index'] = matched_df[f'{F1}_mag'] - matched_df[f'{F2}_mag']
        # Average morphology parameters across both filters
        matched_df['avg_CI'] = (matched_df[f'{F1}_CI'] + matched_df[f'{F2}_CI']) / 2
        matched_df['avg_ellipticity'] = (matched_df[f'{F1}_ellipticity'] + 
                                         matched_df[f'{F2}_ellipticity']) / 2
    
    print(f"Found {len(matched_df)} matched sources")
    return matched_df


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================
# Each filter function takes a DataFrame and returns a filtered copy.
# Global variables used: F1, F2, CI_threshold, ellipticity_threshold

def FILTER_ALL(df):
    """No filtering - just add category label"""
    df['filter_category'] = 'ALL'
    print(f"ALL filter: {len(df)} sources remaining")
    return df


def FILTER_is_extended(df):
    """
    Filter for point sources (non-extended).
    is_extended = 0 for point sources, 1 for extended sources.
    """
    if f'{F1}_is_extended' in df.columns and f'{F2}_is_extended' in df.columns:
        quality_mask = ((df[f'{F1}_is_extended'] == 0) & 
                        (df[f'{F2}_is_extended'] == 0))
        filtered_df = df[quality_mask].copy()
        filtered_df['filter_category'] = 'non_extended'
    else:
        print("Extended source information not available")
        filtered_df = pd.DataFrame(columns=df.columns)
    
    print(f"Extended source filter: {len(filtered_df)} sources remaining")
    return filtered_df


def FILTER_CI(df):
    """
    Filter based on Concentration Index (CI).
    Lower CI = more point-like. Typical range: 1-6.
    Point sources typically have CI < 2.0-2.5.
    """
    if 'avg_CI' not in df.columns:
        print("Concentration Index information not available")
        return pd.DataFrame(columns=df.columns)
    
    quality_mask = ((df['avg_CI'] < CI_threshold) & np.isfinite(df['avg_CI']))
    filtered_df = df[quality_mask].copy()
    filtered_df['filter_category'] = 'CI_filtered'
    
    print(f"CI filter: {len(filtered_df)} sources remaining (CI < {CI_threshold})")
    return filtered_df


def FILTER_ellipticity(df):
    """
    Filter based on ellipticity (roundness).
    Lower ellipticity = more round (point-like).
    """
    if 'avg_ellipticity' not in df.columns:
        print("Ellipticity information not available")
        return pd.DataFrame(columns=df.columns)
    
    quality_mask = ((df['avg_ellipticity'] < ellipticity_threshold) & 
                    np.isfinite(df['avg_ellipticity']))
    filtered_df = df[quality_mask].copy()
    filtered_df['filter_category'] = 'ellipticity_filtered'
    
    print(f"Ellipticity filter: {len(filtered_df)} sources remaining (ellipticity < {ellipticity_threshold})")
    return filtered_df


def FILTER_isochrone_shift(df, isochrone_color, isochrone_mag, shift_amount, width):
    """
    Filter points within a shifted band around a theoretical isochrone.
    Useful for isolating specific stellar populations (e.g., RGB stars).
    """
    if 'color_index' not in df.columns:
        print("Color index not available for isochrone filtering")
        return pd.DataFrame(columns=df.columns)
    
    df_filtered = df.copy()
    
    # Clean and prepare isochrone data
    isochrone_color = np.asarray(isochrone_color).flatten()
    isochrone_mag = np.asarray(isochrone_mag).flatten()
    valid_idx = np.isfinite(isochrone_color) & np.isfinite(isochrone_mag)
    isochrone_color = isochrone_color[valid_idx]
    isochrone_mag = isochrone_mag[valid_idx]
    
    if len(isochrone_color) < 2:
        print("Warning: Insufficient isochrone data points for interpolation")
        return df_filtered
    
    # Sort by magnitude and remove duplicates
    sort_idx = np.argsort(isochrone_mag)
    isochrone_mag_sorted = isochrone_mag[sort_idx]
    isochrone_color_sorted = isochrone_color[sort_idx]
    _, unique_idx = np.unique(isochrone_mag_sorted, return_index=True)
    isochrone_mag_sorted = isochrone_mag_sorted[unique_idx]
    isochrone_color_sorted = isochrone_color_sorted[unique_idx]
    
    # Create interpolation function (color as function of magnitude)
    try:
        interp_func = interp1d(isochrone_mag_sorted, isochrone_color_sorted, 
                              kind='linear', bounds_error=False,
                              fill_value=(isochrone_color_sorted[0], 
                                         isochrone_color_sorted[-1]))
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return df_filtered
    
    mag_min, mag_max = isochrone_mag_sorted.min(), isochrone_mag_sorted.max()
    
    # Apply band filter
    keep_mask = []
    for idx, row in df_filtered.iterrows():
        mag = row[f'{F1}_mag']
        color = row['color_index']
        
        if mag < mag_min:
            expected_color = isochrone_color_sorted[0] + shift_amount
        elif mag > mag_max:
            expected_color = isochrone_color_sorted[-1] + shift_amount
        else:
            expected_color = interp_func(mag) + shift_amount
        
        keep_mask.append(abs(color - expected_color) <= width)
    
    df_filtered = df_filtered[keep_mask]
    df_filtered['filter_category'] = 'isochrone_shifted'
    
    print(f"Isochrone shift filter (shift={shift_amount}, width={width}): {len(df_filtered)} sources remaining")
    return df_filtered


def apply_filter_pipeline(matched_df, dist, filter_list=None, 
                          isochrone_color=None, isochrone_mag=None, 
                          shift_amount=0.0, width=0.5):
    """
    Apply a sequence of filters to isolate point sources.
    
    Parameters:
    -----------
    filter_list : list of str
        Options: 'ALL', 'is_extended', 'CI', 'ellipticity', 'isochrone'
    
    Returns:
    --------
    filtered_dataframes : dict
        DataFrames at each filtering stage for visualization
    """
    if filter_list is None:
        filter_list = ['ALL', 'is_extended', 'CI', 'ellipticity', 'isochrone']
    
    filter_functions = {
        'ALL': FILTER_ALL,
        'is_extended': FILTER_is_extended,
        'CI': FILTER_CI,
        'ellipticity': FILTER_ellipticity,
        'isochrone': lambda df: FILTER_isochrone_shift(
            df, isochrone_color, isochrone_mag, shift_amount, width
        ) if isochrone_color is not None and isochrone_mag is not None else pd.DataFrame(columns=df.columns)
    }
    
    filtered_dataframes = {}
    current_df = matched_df.copy()
    
    print(f"\nApplying filter pipeline in order: {filter_list}")
    print(f"Initial number of sources: {len(matched_df)}")
    
    for filter_name in filter_list:
        if filter_name in filter_functions:
            new_df = filter_functions[filter_name](current_df)
            
            if len(new_df) > 0:
                current_df = new_df
                filtered_dataframes[filter_name] = current_df.copy()
                print(f"After {filter_name} filter: {len(current_df)} sources")
            else:
                print(f"Warning: {filter_name} filter returned empty dataframe - skipping")
                filtered_dataframes[filter_name] = pd.DataFrame(columns=current_df.columns)
        else:
            print(f"Warning: Unknown filter '{filter_name}' - skipping")
    
    filtered_dataframes['final'] = current_df.copy()
    
    print(f"\nFinal number of sources: {len(current_df)}")
    return filtered_dataframes


# ============================================================================
# COORDINATE CONVERSION FUNCTIONS
# ============================================================================

def cart2pol(x, y, center_ra=None, center_dec=None):
    """
    Convert Cartesian coordinates (RA, DEC) to polar coordinates (radius, angle).
    Radius is in degrees, angle in radians.
    """
    if center_ra is None:
        center_ra = np.mean(x)
    if center_dec is None:
        center_dec = np.mean(y)
    
    # Correct for RA compression at given declination
    dx = (x - center_ra) * np.cos(np.deg2rad(center_dec))
    dy = y - center_dec
    
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    return radius, angle


def get_max_radius(df, center_ra, center_dec):
    """Calculate maximum radial distance from center for binning"""
    if len(df) == 0:
        return 0.1  # default fallback
    radius, _ = cart2pol(df['RA'], df['DEC'], center_ra, center_dec)
    return radius.max()


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

# Common style dictionary for filtering stages
STAGE_STYLES = {
    'ALL': {'color': 'gray', 'marker': '.', 'label': 'All matched', 'alpha': 0.4, 'size': 15},
    'non_extended': {'color': 'red', 'marker': 's', 'label': 'Non-extended', 'alpha': 0.5, 'size': 20},
    'CI_filtered': {'color': 'green', 'marker': '^', 'label': 'CI Filtered', 'alpha': 0.6, 'size': 25},
    'ellipticity_filtered': {'color': 'orange', 'marker': 'v', 'label': 'Ellipticity Filtered', 'alpha': 0.7, 'size': 30},
    'isochrone_shifted': {'color': 'brown', 'marker': '*', 'label': 'Isochrone Shifted', 'alpha': 0.8, 'size': 35},
    'final': {'color': 'purple', 'marker': 'D', 'label': 'Final Selection', 'alpha': 0.9, 'size': 40}
}


def plot_cmd_single(filtered_dataframes, isochrone_color, isochrone_mag):
    """
    Figure 1: Color-Magnitude Diagram showing all filtering stages.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for stage, df in filtered_dataframes.items():
        if len(df) > 0 and stage in STAGE_STYLES:
            style = STAGE_STYLES[stage]
            ax.scatter(df['color_index'], df[f'{F1}_mag'],
                      alpha=style['alpha'], s=style['size'], c=style['color'],
                      marker=style['marker'], 
                      label=f"{style['label']} ({len(df)} sources)")
    
    # Plot theoretical isochrone
    ax.scatter(isochrone_color, isochrone_mag, color='b', s=5, 
              label='Theory isochrone: 12000 Myr')
    
    ax.set_xlabel(f'Color Index ({F1} - {F2})')
    ax.set_ylabel(f'{F1} Magnitude')
    ax.set_title(f'{current_galaxy.name}: Color-Magnitude Diagram\n'
                f'Filters: {F1} vs {F2}')
    ax.invert_yaxis()  # Brighter at top
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_cumulative_magnitude(filtered_dataframes):
    """
    Figure 2: Cumulative magnitude distribution for each filtering stage.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for stage, df in filtered_dataframes.items():
        if len(df) > 0 and stage in STAGE_STYLES:
            style = STAGE_STYLES[stage]
            mag_values = df[f'{F1}_mag'].dropna()
            if len(mag_values) > 0:
                sorted_mags = np.sort(mag_values)
                cumul_dist = np.arange(1, len(sorted_mags) + 1) / len(sorted_mags)
                ax.plot(sorted_mags, cumul_dist, color=style['color'],
                       linewidth=2, alpha=style['alpha'],
                       label=f"{style['label']} ({len(df)} sources)")
    
    ax.set_xlabel(f'{F1} Magnitude')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title(f'{current_galaxy.name}: Cumulative Magnitude Distribution\n'
                f'Filter: {F1}')
    ax.invert_xaxis()  # Brighter at right
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_location_cartesian(filtered_dataframes):
    """
    Figure 3: Cartesian RA-DEC location plot with equal aspect ratio.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for stage, df in filtered_dataframes.items():
        if len(df) > 0 and stage in STAGE_STYLES and 'RA' in df.columns:
            style = STAGE_STYLES[stage]
            ax.scatter(df['RA'], df['DEC'],
                      alpha=style['alpha'], s=style['size'], c=style['color'],
                      marker=style['marker'],
                      label=f"{style['label']} ({len(df)} sources)")
    
    # Mark galaxy center
    ax.plot(current_galaxy.RA, current_galaxy.DEC, marker='*', 
           markersize=15, color='gold', markeredgecolor='black',
           label='Galaxy Center')
    
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')
    ax.set_title(f'{current_galaxy.name}: Spatial Distribution\n'
                f'Center: RA={current_galaxy.RA:.5f}, DEC={current_galaxy.DEC:.5f}')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_polar_location(filtered_dataframes):
    """
    Figure 4: Polar coordinates plot with center marked.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    for stage, df in filtered_dataframes.items():
        if len(df) > 0 and stage in STAGE_STYLES and 'RA' in df.columns:
            style = STAGE_STYLES[stage]
            radius, angle = cart2pol(df['RA'], df['DEC'], 
                                     current_galaxy.RA, current_galaxy.DEC)
            ax.scatter(angle, radius, alpha=style['alpha'], s=style['size'], 
                      c=style['color'], marker=style['marker'],
                      label=f"{style['label']} ({len(df)} sources)")
    
    # Mark center (radius = 0)
    ax.scatter(0, 0, marker='*', s=200, c='gold', edgecolor='black',
              label='Galaxy Center', zorder=10)
    
    ax.set_title(f'{current_galaxy.name}: Polar Coordinates\n'
                f'Center at origin')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_polar_histogram_filtered_only(filtered_dataframes):
    """
    Figure 5: Polar histogram (2D density) for final filtered sources only.
    """
    if 'final' not in filtered_dataframes or len(filtered_dataframes['final']) == 0:
        print("No filtered data to plot")
        return
    
    df_final = filtered_dataframes['final']
    radius, angle = cart2pol(df_final['RA'], df_final['DEC'],
                             current_galaxy.RA, current_galaxy.DEC)
    
    # Dynamic binning based on actual data range
    max_r = get_max_radius(df_final, current_galaxy.RA, current_galaxy.DEC)
    n_radial_bins = 20
    n_theta_bins = 30
    
    rbins = np.linspace(0, max_r, n_radial_bins)
    abins = np.linspace(-np.pi, np.pi, n_theta_bins)
    A, R = np.meshgrid(abins, rbins)
    
    # 2D histogram
    hist, _, _ = np.histogram2d(angle, radius, bins=(abins, rbins))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
    pc = ax.pcolormesh(A, R, hist.T, cmap='magma_r')
    ax.set_title(f'{current_galaxy.name}: Filtered Source Density\n'
                f'Final selection: {len(df_final)} sources')
    plt.colorbar(pc, ax=ax, label='Source Count')
    
    plt.tight_layout()
    plt.show()


def plot_polar_histogram_comparison(filtered_dataframes):
    """
    Figure 6: Vertical polar histograms - unfiltered (ALL) on top, filtered (final) on bottom.
    """
    # Create vertical subplots (2 rows, 1 column)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 14), 
                                             subplot_kw={'projection': 'polar'})
    
    # Get dynamic max radius from all data
    if 'ALL' in filtered_dataframes and len(filtered_dataframes['ALL']) > 0:
        all_radius, _ = cart2pol(filtered_dataframes['ALL']['RA'], 
                                 filtered_dataframes['ALL']['DEC'],
                                 current_galaxy.RA, current_galaxy.DEC)
        max_r = all_radius.max()
    else:
        max_r = 0.1  # Default fallback
    
    n_radial_bins = 20
    n_theta_bins = 30
    rbins = np.linspace(0, max_r, n_radial_bins)
    abins = np.linspace(-np.pi, np.pi, n_theta_bins)
    A, R = np.meshgrid(abins, rbins)
    
    # TOP: Unfiltered (ALL)
    if 'ALL' in filtered_dataframes and len(filtered_dataframes['ALL']) > 0:
        df_all = filtered_dataframes['ALL']
        radius_all, angle_all = cart2pol(df_all['RA'], df_all['DEC'],
                                         current_galaxy.RA, current_galaxy.DEC)
        hist_all, _, _ = np.histogram2d(angle_all, radius_all, bins=(abins, rbins))
        pc1 = ax_top.pcolormesh(A, R, hist_all.T, cmap='magma_r')
        ax_top.set_title(f'Unfiltered (ALL) - {len(df_all)} sources', fontsize=12, pad=20)
        plt.colorbar(pc1, ax=ax_top, label='Source Count', fraction=0.046, pad=0.04)
    else:
        ax_top.text(0, 0, 'No unfiltered data', ha='center', va='center', transform=ax_top.transData)
        ax_top.set_title('Unfiltered (ALL) - No Data', fontsize=12, pad=20)
    
    ax_top.grid(True, alpha=0.3)
    
    # BOTTOM: Filtered (final)
    if 'final' in filtered_dataframes and len(filtered_dataframes['final']) > 0:
        df_final = filtered_dataframes['final']
        radius_final, angle_final = cart2pol(df_final['RA'], df_final['DEC'],
                                             current_galaxy.RA, current_galaxy.DEC)
        hist_final, _, _ = np.histogram2d(angle_final, radius_final, bins=(abins, rbins))
        pc2 = ax_bottom.pcolormesh(A, R, hist_final.T, cmap='magma_r')
        ax_bottom.set_title(f'Filtered (Final) - {len(df_final)} sources', fontsize=12, pad=20)
        plt.colorbar(pc2, ax=ax_bottom, label='Source Count', fraction=0.046, pad=0.04)
    else:
        ax_bottom.text(0, 0, 'No filtered data', ha='center', va='center', transform=ax_bottom.transData)
        ax_bottom.set_title('Filtered (Final) - No Data', fontsize=12, pad=20)
    
    ax_bottom.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle(f'{current_galaxy.name}: Polar Density Comparison\nFilters: {F1} & {F2}', 
                 fontsize=16)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    #plt.subplots_adjust(top=0.93, hspace=0.3)
    plt.show()


def plot_radial_density_profile(filtered_dataframes):
    """
    Figure 7: Radial density profile (counts per square degree).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Get max radius from ALL data for consistent binning
    all_radius, _ = cart2pol(filtered_dataframes['ALL']['RA'], 
                             filtered_dataframes['ALL']['DEC'],
                             current_galaxy.RA, current_galaxy.DEC)
    max_r = all_radius.max()
    n_radial_bins = 20
    bin_edges = np.linspace(0, max_r, n_radial_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    for stage, df in filtered_dataframes.items():
        if len(df) > 0 and stage in STAGE_STYLES and 'RA' in df.columns:
            style = STAGE_STYLES[stage]
            radius, _ = cart2pol(df['RA'], df['DEC'],
                                 current_galaxy.RA, current_galaxy.DEC)
            
            hist, _ = np.histogram(radius, bins=bin_edges)
            bin_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
            density = hist / bin_areas  # counts per square degree
            
            ax.plot(bin_centers, density, 
                   marker=style['marker'], markersize=8,
                   color=style['color'], alpha=style['alpha'],
                   linewidth=2, label=f"{style['label']} ({len(df)} sources)")
    
    ax.set_xlabel('Radius (degrees)')
    ax.set_ylabel('Surface Density (counts/deg²)')
    ax.set_title(f'{current_galaxy.name}: Radial Density Profile\n'
                f'Filters: {F1} & {F2}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# FILEPATH CONFIGURATION - RELATIVE TO SCRIPT LOCATION
# ============================================================================
# Get the directory where this script is located (should be HALOS directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HALOS_DIR = SCRIPT_DIR

# Define subdirectory paths relative to HALOS
THEORY_ISOCHRONES_DIR = os.path.join(HALOS_DIR, "Theory Isochrones")
JWST_DATA_DIR = os.path.join(HALOS_DIR, "JWST_Data")
INFO_CSV_PATH = os.path.join(HALOS_DIR, "JWST_GLX_INFO_FULL_DRAFT.csv")
ISOCHRONE_FILE = os.path.join(THEORY_ISOCHRONES_DIR, "12000JWST_VEGA_11_13.isc_jwst-nircam_PL")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Load theoretical isochrone (12 Gyr BaSTI model)
df_12000 = pd.read_csv(ISOCHRONE_FILE, sep='\s+')

# Load galaxy catalog
data_types = {
    'NAME': str, 'RA': float, 'DEC': float, 'DISTANCE': float,
    'FILTER1': str, 'FILTER2': str
}
INFO = pd.read_csv(INFO_CSV_PATH, dtype=data_types)

# Create galaxy objects (path will be constructed per galaxy)
path = JWST_DATA_DIR  # Base path, then append galaxy name when needed
glx = np.zeros(len(INFO), dtype=GALAXY)
for i in range(len(INFO)):
    glx[i] = GALAXY(INFO.NAME[i], INFO.RA[i], INFO.DEC[i], INFO.DISTANCE[i])
    print(str(i) + ": " + glx[i].name)

# ============================================================================
# USER CONTROLS - ADJUST THESE PARAMETERS
# ============================================================================
num = 2                     # Index of galaxy to process
match_radius_arcsec = 0.03   # Matching radius between filters (arcseconds)
CI_threshold = 2.0           # Concentration Index threshold (lower = more point-like)
ellipticity_threshold = 0.3  # Ellipticity threshold (lower = rounder)

# ============================================================================
# PROCESS SELECTED GALAXY
# ============================================================================
current_galaxy = glx[num]
F1 = INFO.FILTER1[num]
F2 = INFO.FILTER2[num]

print(f"\n{'='*60}")
print(f"Processing galaxy: {current_galaxy.name}")
print(f"Filters: {F1} and {F2}")
print(f"Distance: {current_galaxy.dist} Mpc")
print(f"{'='*60}\n")

# Change to galaxy data directory - FIXED VERSION
galaxy_path = os.path.join(path, current_galaxy.name)
print(f"Changing to directory: {galaxy_path}")
os.chdir(galaxy_path)

# Load JWST catalogs
all_filters = load_all_filters()

if all_filters:
    print(f"\nSuccessfully loaded {len(all_filters)} filters: {list(all_filters.keys())}")
    
    # Match sources between filters
    matched_data = match_sources_across_filters(all_filters)
    
    # Calculate apparent magnitudes for theoretical isochrone
    app_A_f1, app_A_f2 = app_mag(12000, df_12000[F1], df_12000[F2], current_galaxy.dist)
    idx_A = app_A_f1 - app_A_f2  # Color index
    
    # Set plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    if matched_data is not None and len(matched_data) > 0:
        
        # Define filter pipeline order
        custom_filter_pipeline = [
            'ALL',           # Start with all sources
            'is_extended',   # Remove extended sources
            'CI',            # Apply Concentration Index filter (point source selection)
            'ellipticity',   # Apply ellipticity filter (roundness)
            'isochrone'      # Match to theoretical isochrone
        ]
        
        # Apply all filters
        filtered_dataframes = apply_filter_pipeline(
            matched_data, 
            current_galaxy.dist,
            filter_list=custom_filter_pipeline,
            isochrone_color=idx_A,
            isochrone_mag=app_A_f2,
            shift_amount=-0.0,  # Horizontal shift for isochrone matching
            width=0.3          # Width of matching band
        )
        
        # ====================================================================
        # GENERATE ALL PLOTS
        # ====================================================================
        
        # Figure 1: Color-Magnitude Diagram
        plot_cmd_single(filtered_dataframes, idx_A, app_A_f2)
        
        # Figure 2: Cumulative Magnitude Distribution
        plot_cumulative_magnitude(filtered_dataframes)
        
        # Figure 3: Cartesian RA-DEC location plot
        plot_location_cartesian(filtered_dataframes)
        
        # Figure 4: Polar coordinates with center marked
        plot_polar_location(filtered_dataframes)
        
        # Figure 5: Polar histogram (filtered only)
        plot_polar_histogram_filtered_only(filtered_dataframes)
        
        # Figure 6: Polar histogram comparison (unfiltered vs filtered)
        plot_polar_histogram_comparison(filtered_dataframes)
        
        # Figure 7: Radial density profile
        plot_radial_density_profile(filtered_dataframes)
        
    else:
        print("No matched data found or matched data is empty")
else:
    print("Failed to load filter catalogs")
