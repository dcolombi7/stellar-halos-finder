# -*- coding: utf-8 -*-
"""
JWST Galaxy Analysis Pipeline
Author: Daniela Sofia Colombi
University of Maryland - College Park
May 15, 2026

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
from shapely.geometry import Polygon, Point
import math
from scipy.cluster.vq import kmeans
import matplotlib.image as mpimg


# ============================================================================
# CONSTANTS & GLOBAL SETTINGS
# ============================================================================
# JWST NIRCAM filters (both short and long wavelength channels)
NIRCAM_filters = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
                'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W',
                'F460M', 'F466N', 'F470N', 'F480M']

# NIRCAM detector geometry constants
NIRCAM_DETECTOR_SIZE_ARCSEC = 132.0  # 2.2 arcminutes = 132 arcseconds
NIRCAM_GAP_ARCSEC = 44.0  # 44 arcsecond gap between detectors


# ============================================================================
# NIRCAM FOOTPRINT DETECTION FROM DATA
# ============================================================================

class NIRCAMFootprint:
    """NIRCAM footprint detected directly from source distribution"""
    def __init__(self, df, galaxy_ra, galaxy_dec, percentile_threshold=95):
        """
        Detect footprint from data.
        
        Parameters:
        -----------
        df : DataFrame
            Unfiltered source catalog with RA, DEC columns
        galaxy_ra, galaxy_dec : float
            Galaxy center coordinates (for reference)
        percentile_threshold : float
            For edge detection
        """
        self.galaxy_ra = galaxy_ra
        self.galaxy_dec = galaxy_dec
        
        # Detect footprint from data - using df['RA'] and df['DEC'] like rest of code
        self.detector_centers, self.detector_sizes, self.position_angle = \
            self._detect_footprint_from_data(df, galaxy_ra, galaxy_dec, percentile_threshold)
        
        # Create Shapely polygons for each detector
        self.detectors = []
        self._create_detector_polygons_from_centers()
        
        print(f"Detected {len(self.detectors)} NIRCAM detectors")
        print(f"Position angle: {self.position_angle:.1f} degrees")
        for i, (center, size) in enumerate(zip(self.detector_centers, self.detector_sizes)):
            print(f"  Detector {chr(65+i)}: center=({center[0]:.5f}, {center[1]:.5f}), "
                  f"size={size[0]*3600:.1f}\" x {size[1]*3600:.1f}\"")
    
    def _detect_footprint_from_data(self, df, center_ra, center_dec, percentile_threshold=95):
        """
        Detect NIRCAM detector placement by finding the bounding boxes of source distributions.
        """
        # Extract RA and DEC arrays
        ra_values = df['RA'].values
        dec_values = df['DEC'].values
        
        # Convert to relative coordinates (arcseconds)
        cos_dec = np.cos(np.radians(center_dec))
        x_arcsec = (ra_values - center_ra) * 3600.0 * cos_dec
        y_arcsec = (dec_values - center_dec) * 3600.0
        
        # Known fixed geometry
        detector_size_arcsec = NIRCAM_DETECTOR_SIZE_ARCSEC
        gap_arcsec = NIRCAM_GAP_ARCSEC
        half_detector = detector_size_arcsec / 2.0
        
        # Use k-means to find two cluster centers
        coords = np.column_stack((x_arcsec, y_arcsec))
        
        # Run k-means to find 2 cluster centers
        centroids, _ = kmeans(coords, 2)
        
        # Sort by x-coordinate
        centroids = centroids[np.argsort(centroids[:, 0])]
        
        # Assign each point to its nearest centroid to get the two clusters
        from scipy.cluster.vq import vq
        labels, _ = vq(coords, centroids)
        
        # Get the bounding box of each cluster (the actual extent of sources)
        cluster_0_points = coords[labels == 0]
        cluster_1_points = coords[labels == 1]
        
        # For each cluster, find the min/max of x and y
        x_min_0, x_max_0 = cluster_0_points[:, 0].min(), cluster_0_points[:, 0].max()
        y_min_0, y_max_0 = cluster_0_points[:, 1].min(), cluster_0_points[:, 1].max()
        
        x_min_1, x_max_1 = cluster_1_points[:, 0].min(), cluster_1_points[:, 0].max()
        y_min_1, y_max_1 = cluster_1_points[:, 1].min(), cluster_1_points[:, 1].max()
        
        # Calculate the center of each bounding box (should be the detector center)
        center_x_0 = (x_min_0 + x_max_0) / 2
        center_y_0 = (y_min_0 + y_max_0) / 2
        
        center_x_1 = (x_min_1 + x_max_1) / 2
        center_y_1 = (y_min_1 + y_max_1) / 2
        
        # Sort by x-coordinate
        if center_x_0 > center_x_1:
            center_x_0, center_x_1 = center_x_1, center_x_0
            center_y_0, center_y_1 = center_y_1, center_y_0
        
        detector_centers_arcsec = [(center_x_0, center_y_0), (center_x_1, center_y_1)]
        
        # Calculate orientation angle
        dx = center_x_1 - center_x_0
        dy = center_y_1 - center_y_0
        position_angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate the actual separation between bounding box centers
        actual_separation = np.sqrt(dx**2 + dy**2)
        expected_separation = detector_size_arcsec + gap_arcsec
        
        print(f"Detector 0 bounds: x=[{x_min_0:.1f}, {x_max_0:.1f}], y=[{y_min_0:.1f}, {y_max_0:.1f}]")
        print(f"Detector 1 bounds: x=[{x_min_1:.1f}, {x_max_1:.1f}], y=[{y_min_1:.1f}, {y_max_1:.1f}]")
        print(f"Detected separation: {actual_separation:.1f}\", expected: {expected_separation}\"")
        print(f"Position angle: {position_angle:.1f}°")
        
        # Optional: Apply a small manual offset if needed (adjust these values)
        # These are in arcseconds - adjust based on your image
        manual_offset_x = 0  # Shift left/right (positive = right)
        manual_offset_y = 0  # Shift up/down (positive = up)
        
        # Apply manual offset if specified
        if manual_offset_x != 0 or manual_offset_y != 0:
            detector_centers_arcsec = [
                (detector_centers_arcsec[0][0] + manual_offset_x, 
                 detector_centers_arcsec[0][1] + manual_offset_y),
                (detector_centers_arcsec[1][0] + manual_offset_x, 
                 detector_centers_arcsec[1][1] + manual_offset_y)
            ]
            print(f"Applied manual offset: ({manual_offset_x}, {manual_offset_y})\"")
        
        # Convert to sky coordinates
        detector_centers = []
        detector_sizes = []
        
        for x_center_arcsec, y_center_arcsec in detector_centers_arcsec:
            ra_center = center_ra + x_center_arcsec / 3600.0 / cos_dec
            dec_center = center_dec + y_center_arcsec / 3600.0
            detector_centers.append((ra_center, dec_center))
            detector_sizes.append((detector_size_arcsec / 3600.0, detector_size_arcsec / 3600.0))
        
        return detector_centers, detector_sizes, position_angle
        
    def _create_detector_polygons_from_centers(self):
        """Create polygons from detected detector centers and sizes"""
        cos_dec = np.cos(np.radians(self.galaxy_dec))
        angle_rad = math.radians(self.position_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        for (center_ra, center_dec), (width_deg, height_deg) in \
            zip(self.detector_centers, self.detector_sizes):
            
            half_w = width_deg / 2.0
            half_h = height_deg / 2.0
            
            corners = []
            for dx, dy in [(-half_w, -half_h), (half_w, -half_h), 
                          (half_w, half_h), (-half_w, half_h)]:
                # Rotate
                x_rot = dx * cos_angle - dy * sin_angle
                y_rot = dx * sin_angle + dy * cos_angle
                
                # Convert to sky coordinates
                ra = center_ra + x_rot / cos_dec
                dec = center_dec + y_rot
                corners.append((ra, dec))
            
            self.detectors.append(Polygon(corners))
    
    def contains_point(self, ra, dec):
        """Check if point falls within any detector"""
        point = Point(ra, dec)
        return any(detector.contains(point) for detector in self.detectors)
    
    def get_footprint_patch(self, ax, color='gray', alpha=0.3, label='NIRCAM Footprint'):
        """Add footprint visualization to matplotlib axis"""
        for i, detector in enumerate(self.detectors):
            x, y = detector.exterior.xy
            ax.fill(x, y, alpha=alpha, fc=color, ec='black', linewidth=2,
                   label=label if i == 0 else None)
    
    def get_bin_coverage_fraction(self, r_min, r_max, a_min, a_max, n_samples=20):
        """Calculate fraction of polar bin within footprint"""
        np.random.seed(42)
        
        r_samples = np.random.uniform(r_min, r_max, n_samples)
        a_samples = np.random.uniform(a_min, a_max, n_samples)
        
        inside_count = 0
        
        for r in r_samples:
            for a in a_samples:
                ra_sample = self.galaxy_ra + r * np.cos(a) / np.cos(np.radians(self.galaxy_dec))
                dec_sample = self.galaxy_dec + r * np.sin(a)
                
                if self.contains_point(ra_sample, dec_sample):
                    inside_count += 1
        
        return inside_count / (n_samples * n_samples)


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

class GALAXY:
    """Container for galaxy properties from MAST catalog"""
    def __init__(self, name, RA, DEC, dist, hlr):
        self.name = name      # Galaxy name (used for folder access)
        self.RA = RA          # Right Ascension (degrees)
        self.DEC = DEC        # Declination (degrees)
        self.dist = dist      # Distance to galaxy (Mpc)
        self.hlr = hlr        # Halflight radius of galaxy (kpc)


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
    """Load catalogs for two specified filters from current directory."""
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
    """Match sources between two filters based on RA/DEC coordinates."""
    if F1 not in filter_data or F2 not in filter_data:
        print(f"Error: {F1} or {F2} not found in filter data")
        return None
    
    df1 = filter_data[F1].copy()
    df2 = filter_data[F2].copy()
    
    match_radius_deg = match_radius_arcsec / 3600.0
    
    print(f"Matching sources between {F1} and {F2}")
    print(f"{F1}: {len(df1)} sources, {F2}: {len(df2)} sources")
    
    matched_sources = []
    
    for idx1, row1 in df1.iterrows():
        ra1, dec1 = row1['sky_centroid.ra'], row1['sky_centroid.dec']
        
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
        matched_df['color_index'] = matched_df[f'{F1}_mag'] - matched_df[f'{F2}_mag']
        matched_df['avg_CI'] = (matched_df[f'{F1}_CI'] + matched_df[f'{F2}_CI']) / 2
        matched_df['avg_ellipticity'] = (matched_df[f'{F1}_ellipticity'] + 
                                         matched_df[f'{F2}_ellipticity']) / 2
    
    print(f"Found {len(matched_df)} matched sources")
    return matched_df


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def FILTER_ALL(df):
    """No filtering - just add category label"""
    df['filter_category'] = 'ALL'
    print(f"ALL filter: {len(df)} sources remaining")
    return df


def FILTER_limmag(df):
    """Filter using qualitative limiting magnitude."""
    if f'{F1}_mag' in df.columns and f'{F2}_mag' in df.columns:
        quality_mask = ((df[f'{F1}_mag'] < limiting_magnitude) & 
                        (df[f'{F2}_mag'] < limiting_magnitude))
        filtered_df = df[quality_mask].copy()
        filtered_df['filter_category'] = 'limmag'
    else:
        print("Magnitude information not available")
        filtered_df = pd.DataFrame(columns=df.columns)
    
    print(f"Limiting magnitude filter: {len(filtered_df)} sources remaining")
    return filtered_df


def FILTER_is_extended(df):
    """Filter for point sources (non-extended)."""
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
    """Filter based on Concentration Index (CI)."""
    if 'avg_CI' not in df.columns:
        print("Concentration Index information not available")
        return pd.DataFrame(columns=df.columns)
    
    quality_mask = ((df['avg_CI'] < CI_threshold) & np.isfinite(df['avg_CI']))
    filtered_df = df[quality_mask].copy()
    filtered_df['filter_category'] = 'CI_filtered'
    
    print(f"CI filter: {len(filtered_df)} sources remaining (CI < {CI_threshold})")
    return filtered_df


def FILTER_ellipticity(df):
    """Filter based on ellipticity (roundness)."""
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
    """Filter points within a shifted band around a theoretical isochrone."""
    if 'color_index' not in df.columns:
        print("Color index not available for isochrone filtering")
        return pd.DataFrame(columns=df.columns)
    
    df_filtered = df.copy()
    
    isochrone_color = np.asarray(isochrone_color).flatten()
    isochrone_mag = np.asarray(isochrone_mag).flatten()
    valid_idx = np.isfinite(isochrone_color) & np.isfinite(isochrone_mag)
    isochrone_color = isochrone_color[valid_idx]
    isochrone_mag = isochrone_mag[valid_idx]
    
    if len(isochrone_color) < 2:
        print("Warning: Insufficient isochrone data points for interpolation")
        return df_filtered
    
    sort_idx = np.argsort(isochrone_mag)
    isochrone_mag_sorted = isochrone_mag[sort_idx]
    isochrone_color_sorted = isochrone_color[sort_idx]
    _, unique_idx = np.unique(isochrone_mag_sorted, return_index=True)
    isochrone_mag_sorted = isochrone_mag_sorted[unique_idx]
    isochrone_color_sorted = isochrone_color_sorted[unique_idx]
    
    try:
        interp_func = interp1d(isochrone_mag_sorted, isochrone_color_sorted, 
                              kind='linear', bounds_error=False,
                              fill_value=(isochrone_color_sorted[0], 
                                         isochrone_color_sorted[-1]))
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return df_filtered
    
    mag_min, mag_max = isochrone_mag_sorted.min(), isochrone_mag_sorted.max()
    
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
    """Apply a sequence of filters to isolate point sources."""
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


def apply_nircam_mask_to_histogram(hist, rbins, abins, df_unfiltered, 
                                   galaxy_ra, galaxy_dec, min_coverage=1.0):
    """
    Mask histogram bins that don't meet coverage criteria within NIRCAM footprint.
    """
    footprint = NIRCAMFootprint(df_unfiltered, galaxy_ra, galaxy_dec)
    
    n_radial_bins = len(rbins) - 1
    n_theta_bins = len(abins) - 1
    
    coverage_mask = np.zeros((n_theta_bins, n_radial_bins), dtype=bool)
    n_samples = 15
    
    print(f"Masking histogram bins: {n_theta_bins} theta bins × {n_radial_bins} radial bins")
    
    for i_theta in range(n_theta_bins):
        a_min = abins[i_theta]
        a_max = abins[i_theta + 1]
        
        for i_r in range(n_radial_bins):
            r_min = rbins[i_r]
            r_max = rbins[i_r + 1]
            
            coverage_fraction = footprint.get_bin_coverage_fraction(r_min, r_max, a_min, a_max, n_samples)
            
            if coverage_fraction >= min_coverage:
                coverage_mask[i_theta, i_r] = True
        
        if (i_theta + 1) % 5 == 0:
            print(f"  Processed {i_theta + 1}/{n_theta_bins} theta bins")
    
    masked_hist = np.ma.masked_where(~coverage_mask, hist)
    
    n_masked = np.sum(~coverage_mask)
    n_total = n_theta_bins * n_radial_bins
    print(f"Masked {n_masked} of {n_total} bins ({100 * n_masked / n_total:.1f}%)")
    
    return masked_hist, coverage_mask


# ============================================================================
# COORDINATE CONVERSION FUNCTIONS
# ============================================================================

def cart2pol(x, y, center_ra=None, center_dec=None):
    """Convert Cartesian coordinates (RA, DEC) to polar coordinates."""
    if center_ra is None:
        center_ra = np.mean(x)
    if center_dec is None:
        center_dec = np.mean(y)
    
    dx = (x - center_ra) * np.cos(np.deg2rad(center_dec))
    dy = y - center_dec
    
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    return radius, angle


def get_max_radius(df, center_ra, center_dec):
    """Calculate maximum radial distance from center for binning"""
    if len(df) == 0:
        return 0.1
    radius, _ = cart2pol(df['RA'], df['DEC'], center_ra, center_dec)
    return radius.max()


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

STAGE_STYLES = {
    'ALL': {'color': 'gray', 'marker': '.', 'label': 'All matched', 'alpha': 0.4, 'size': 30},
    'resolved': {'color': 'black', 'marker': '.', 'label': 'Resolved', 'alpha': 0.4, 'size': 30},
    'is_extended': {'color': 'red', 'marker': 's', 'label': 'Non-extended', 'alpha': 0.5, 'size': 25},
    'CI': {'color': 'green', 'marker': '^', 'label': 'CI Filtered', 'alpha': 0.6, 'size': 20},
    'ellipticity': {'color': 'blue', 'marker': 'v', 'label': 'Ellipticity Filtered', 'alpha': 0.7, 'size': 15},
    'isochrone': {'color': 'brown', 'marker': '*', 'label': 'Isochrone Shifted', 'alpha': 0.8, 'size': 10},
    
    'final': {'color': 'purple', 'marker': '.', 'label': 'Old stars', 'alpha': 0.5, 'size': 10},
    'galaxies': {'color': 'orange', 'marker': '.', 'label': 'Galaxies', 'alpha': 0.5, 'size': 10},
    'young_stars': {'color': 'cyan', 'marker': '.', 'label': 'Young stars', 'alpha': 0.5, 'size': 10},
}

def get_last_photometric_filter(filter_list):
    
    #Identify the last JWST photometric filter in the pipeline.
    #Photometric filters are: 'is_extended', 'CI', 'ellipticity'
    
    photometric_filters = ['is_extended', 'CI', 'ellipticity']
    last_filter = None
    
    for filter_name in filter_list:
        if filter_name in photometric_filters:
            last_filter = filter_name
    
    if last_filter is None:
        raise ValueError("No JWST photometric filters selected. Please select at least one JWST photometric filter to filter out point sources.")
    
    return last_filter


def create_derived_dataframes(filtered_dataframes, last_photometric_filter):
    
    #Create derived dataframes:
    #- 'galaxies': ALL minus only_stars (last photometric filter stage)
    #- 'young_stars': only_stars minus final
    
    derived_dataframes = filtered_dataframes.copy()
    
    # Get the 'only stars' dataframe (last photometric filter stage)
    if last_photometric_filter not in filtered_dataframes:
        print(f"Warning: {last_photometric_filter} not found in filtered dataframes")
        return derived_dataframes
    
    only_stars_df = filtered_dataframes[last_photometric_filter]
    all_df = filtered_dataframes.get('ALL', pd.DataFrame())
    final_df = filtered_dataframes.get('final', pd.DataFrame())
    
    # Create 'galaxies' (ALL minus only_stars)
    if len(all_df) > 0 and len(only_stars_df) > 0:
        # Use source_id if available, otherwise use RA/DEC matching
        if 'source_id' in all_df.columns and 'source_id' in only_stars_df.columns:
            stars_indices = set(only_stars_df['source_id'].values)
            galaxies_mask = ~all_df['source_id'].isin(stars_indices)
            galaxies_df = all_df[galaxies_mask].copy()
        else:
            # Match based on RA/DEC with tolerance
            from sklearn.neighbors import NearestNeighbors
            stars_coords = only_stars_df[['RA', 'DEC']].values
            all_coords = all_df[['RA', 'DEC']].values
            
            nn = NearestNeighbors(radius=match_radius_arcsec / 3600.0)
            nn.fit(stars_coords)
            matches = nn.radius_neighbors(all_coords, return_distance=False)
            galaxies_mask = np.array([len(m) == 0 for m in matches])
            galaxies_df = all_df[galaxies_mask].copy()
        
        galaxies_df['filter_category'] = 'galaxies'
        derived_dataframes['galaxies'] = galaxies_df
        print(f"Created 'galaxies' dataframe: {len(galaxies_df)} sources (ALL minus {last_photometric_filter})")
    else:
        print("Could not create 'galaxies' dataframe - missing ALL or only_stars data")
        derived_dataframes['galaxies'] = pd.DataFrame()
    
    # Create 'young_stars' (only_stars minus final)
    if len(only_stars_df) > 0 and len(final_df) > 0:
        if 'source_id' in only_stars_df.columns and 'source_id' in final_df.columns:
            final_indices = set(final_df['source_id'].values)
            young_stars_mask = ~only_stars_df['source_id'].isin(final_indices)
            young_stars_df = only_stars_df[young_stars_mask].copy()
        else:
            # Match based on RA/DEC with tolerance
            from sklearn.neighbors import NearestNeighbors
            final_coords = final_df[['RA', 'DEC']].values
            stars_coords = only_stars_df[['RA', 'DEC']].values
            
            nn = NearestNeighbors(radius=match_radius_arcsec / 3600.0)
            nn.fit(final_coords)
            matches = nn.radius_neighbors(stars_coords, return_distance=False)
            young_stars_mask = np.array([len(m) == 0 for m in matches])
            young_stars_df = only_stars_df[young_stars_mask].copy()
        
        young_stars_df['filter_category'] = 'young_stars'
        derived_dataframes['young_stars'] = young_stars_df
        print(f"Created 'young_stars' dataframe: {len(young_stars_df)} sources ({last_photometric_filter} minus final)")
    else:
        print("Could not create 'young_stars' dataframe - missing only_stars or final data")
        derived_dataframes['young_stars'] = pd.DataFrame()
    
    return derived_dataframes

def get_stages_to_plot(filtered_dataframes, last_photometric_filter):
    
    #Returns list of (stage_name, dataframe) tuples for the three stages to plot:
    #1. galaxies (ALL minus only_stars)
    #2. young_stars (only_stars minus final)
    #3. final (only old stars)
    
    stages_to_plot = []
    
    # Stage 1: galaxies
    if 'galaxies' in filtered_dataframes and len(filtered_dataframes['galaxies']) > 0:
        stages_to_plot.append(('galaxies', filtered_dataframes['galaxies']))
    else:
        print("Warning: 'galaxies' dataframe not available for plotting")
    
    # Stage 2: young_stars
    if 'young_stars' in filtered_dataframes and len(filtered_dataframes['young_stars']) > 0:
        stages_to_plot.append(('young_stars', filtered_dataframes['young_stars']))
    else:
        print("Warning: 'young_stars' dataframe not available for plotting")
    
    # Stage 3: final (old stars)
    if 'final' in filtered_dataframes and len(filtered_dataframes['final']) > 0:
        stages_to_plot.append(('final', filtered_dataframes['final']))
    else:
        print("Warning: 'final' dataframe not available for plotting")
    
    return stages_to_plot


def plot_cmd_single(filtered_dataframes, isochrone_color, isochrone_mag, last_photometric_filter):
    """Figure 1: Color-Magnitude Diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
     
    for stage, df in stages_to_plot:
        if len(df) > 0 and stage in STAGE_STYLES:
            style = STAGE_STYLES[stage]
            label_text = f"{style['label']} ({len(df)} sources)"
            
            ax.scatter(df['color_index'], df[f'{F1}_mag'],
                      alpha=style['alpha'], s=style['size'], c=style['color'],
                      marker=style['marker'], 
                      label=label_text)
    
    ax.scatter(isochrone_color + shift_i, isochrone_mag, color='r', s=3, 
              label=f'Theory isochrone: {age_i} Myr')
    
    ax.set_xlabel(f'Color Index ({F1} - {F2})')
    ax.set_ylabel(f'{F1} Magnitude')
    ax.set_title(f'{current_galaxy.name}: Color-Magnitude Diagram\n'
                f'Filters: {F1} vs {F2}\n'
                f'Photometric filter: {last_photometric_filter}')
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("CMD.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_cumulative_magnitude(filtered_dataframes, last_photometric_filter):
    """Figure 2: Cumulative magnitude distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
     
    for stage, df in stages_to_plot:
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
                f'Filter: {F1}\n'
                f'Photometric filter: {last_photometric_filter}')
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("cumulative_magnitude.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_location_cartesian(filtered_dataframes, last_photometric_filter):
    """Figure 3: Cartesian RA-DEC location plot with detected footprint."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
    unfiltered_df = filtered_dataframes.get('ALL', pd.DataFrame())
    
    for stage, df in stages_to_plot:
        if len(df) > 0 and stage in STAGE_STYLES and 'RA' in df.columns:
            style = STAGE_STYLES[stage]
            ax.scatter(df['RA'], df['DEC'],
                      alpha=style['alpha'], s=style['size'], c=style['color'],
                      marker=style['marker'],
                      label=f"{style['label']} ({len(df)} sources)")
    
    ax.plot(current_galaxy.RA, current_galaxy.DEC, marker='*', 
           markersize=15, color='gold', markeredgecolor='black',
           label='Galaxy Center')
    
    if len(unfiltered_df) > 0:
        footprint = NIRCAMFootprint(unfiltered_df, current_galaxy.RA, current_galaxy.DEC)
        footprint.get_footprint_patch(ax, color='gray', alpha=0.2, label='NIRCAM Footprint')
    
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')
    ax.set_title(f'{current_galaxy.name}: Spatial Distribution with Detected NIRCAM Footprint\n'
                f'Photometric filter: {last_photometric_filter}')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("cartesian_location.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_polar_location(filtered_dataframes, last_photometric_filter=None):
    """Figure 4: Polar coordinates plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
     
    for stage, df in stages_to_plot:
        if len(df) > 0 and stage in STAGE_STYLES and 'RA' in df.columns:
            style = STAGE_STYLES[stage]
            radius, angle = cart2pol(df['RA'], df['DEC'], 
                                     current_galaxy.RA, current_galaxy.DEC)
            ax.scatter(angle, radius, alpha=style['alpha'], s=style['size'], 
                      c=style['color'], marker=style['marker'],
                      label=f"{style['label']} ({len(df)} sources)")
    
    ax.scatter(0, 0, marker='*', s=200, c='gold', edgecolor='black',
              label='Galaxy Center', zorder=10)
    
    if last_photometric_filter:
        filter_desc = f" (photometric filter: {last_photometric_filter})"
    else:
        filter_desc = ""
    
    ax.set_title(f'{current_galaxy.name}: Polar Coordinates\n'
                f'Center at origin{filter_desc}')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("polar_location.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_polar_histogram_filtered_only(filtered_dataframes, min_coverage=1.0, last_photometric_filter=None):
    """Figure 5: Polar histogram masked to detected NIRCAM footprint."""
    # Use the final stage (old stars) for this plot
    if 'final' not in filtered_dataframes or len(filtered_dataframes['final']) == 0:
        print("No final (old stars) data to plot")
        return
    
    df_final = filtered_dataframes['final']
    df_all = filtered_dataframes.get('ALL', pd.DataFrame())
    
    radius, angle = cart2pol(df_final['RA'], df_final['DEC'],
                             current_galaxy.RA, current_galaxy.DEC)
    
    max_r = get_max_radius(df_all if len(df_all) > 0 else df_final, 
                          current_galaxy.RA, current_galaxy.DEC)
    gap_deg = NIRCAM_GAP_ARCSEC / 3600.0
    radial_step = gap_deg / step_factor
    n_radial_bins = max(10, int(np.ceil(max_r / radial_step)))
    rbins = np.linspace(0, max_r, n_radial_bins + 1)
    abins = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
    
    hist, _, _ = np.histogram2d(angle, radius, bins=(abins, rbins))
    
    if len(df_all) > 0:
        masked_hist, _ = apply_nircam_mask_to_histogram(
            hist, rbins, abins, df_all, current_galaxy.RA, current_galaxy.DEC, min_coverage
        )
    else:
        masked_hist = hist
    
    A, R = np.meshgrid(abins, rbins)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
    pc = ax.pcolormesh(A, R, masked_hist.T, cmap='magma_r')
    ax.set_title(f'{current_galaxy.name}: Old Stars Density\n'
                f'Masked to detected NIRCAM footprint\n'
                f'Final selection (old stars): {len(df_final)} sources\n'
                f'Bins require {min_coverage*100:.0f}% coverage')
    plt.colorbar(pc, ax=ax, label='Source Count')
    
    plt.tight_layout()
    plt.savefig("polar_histogram.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_polar_histogram_comparison(filtered_dataframes, min_coverage=1.0, last_photometric_filter=None):
    """Figure 6: 2x2 polar histograms comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), 
                             subplot_kw={'projection': 'polar'})
    ax_tl = axes[0, 0]  # top left
    ax_tr = axes[0, 1]  # top right
    ax_bl = axes[1, 0]  # bottom left
    ax_br = axes[1, 1]  # bottom right
    
    # Get the three derived stages to plot
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
    
    # Use ALL for max radius calculation
    df_all = filtered_dataframes.get('ALL', pd.DataFrame())
    
    if len(df_all) > 0:
        all_radius, _ = cart2pol(df_all['RA'], df_all['DEC'],
                                 current_galaxy.RA, current_galaxy.DEC)
        max_r = all_radius.max()
    else:
        max_r = 0.1
    
    gap_deg = NIRCAM_GAP_ARCSEC / 3600.0
    radial_step = gap_deg / step_factor
    n_radial_bins = max(10, int(np.ceil(max_r / radial_step)))
    rbins = np.linspace(0, max_r, n_radial_bins + 1)
    abins = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
    
    A, R = np.meshgrid(abins, rbins)
    
    # Helper function to create polar histogram
    def create_polar_histogram(ax, df, title, label):
        if len(df) > 0:
            radius, angle = cart2pol(df['RA'], df['DEC'],
                                     current_galaxy.RA, current_galaxy.DEC)
            hist, _, _ = np.histogram2d(angle, radius, bins=(abins, rbins))
            masked_hist, _ = apply_nircam_mask_to_histogram(
                hist, rbins, abins, df_all if len(df_all) > 0 else df,
                current_galaxy.RA, current_galaxy.DEC, min_coverage
            )
            pc = ax.pcolormesh(A, R, masked_hist.T, cmap='magma_r')
            ax.set_title(title, fontsize=12, pad=20)
            plt.colorbar(pc, ax=ax, label=label, fraction=0.046, pad=0.08)
        else:
            ax.text(0, 0, 'No data available', ha='center', va='center', 
                   transform=ax.transData, fontsize=12)
            ax.set_title(f'{title} - No Data', fontsize=12, pad=20)
        ax.grid(True, alpha=0.3)
    
    # Top Left: resolved
    if len(stages_to_plot) >= 0:
        stage_name, stage_df = stages_to_plot[0]
        if stage_name == 'unresolved' and len(stage_df) > 0:
            create_polar_histogram(ax_tl, stage_df,
                                  f'{STAGE_STYLES[stage_name]["label"]} - {len(stage_df)} sources',
                                  'Source Count')
        else:
            ax_tl.text(0, 0, 'No unresolved sources data available', ha='center', va='center', 
                      transform=ax_tl.transData, fontsize=12)
            ax_tl.set_title('Unresolved - No Data', fontsize=12, pad=20)
            ax_tl.grid(True, alpha=0.3)
    else:
        ax_tl.text(0, 0, 'No unresolved sources data available', ha='center', va='center', 
                  transform=ax_bl.transData, fontsize=12)
        ax_tl.set_title('Unresolved - No Data', fontsize=12, pad=20)
        ax_tl.grid(True, alpha=0.3)
    
    # Bottom Left: Galaxies only (ALL minus only_stars)
    if len(stages_to_plot) >= 1:
        stage_name, stage_df = stages_to_plot[1]
        if stage_name == 'galaxies' and len(stage_df) > 0:
            create_polar_histogram(ax_bl, stage_df,
                                  f'{STAGE_STYLES[stage_name]["label"]} - {len(stage_df)} sources',
                                  'Source Count')
        else:
            ax_bl.text(0, 0, 'No galaxies data available', ha='center', va='center', 
                      transform=ax_bl.transData, fontsize=12)
            ax_bl.set_title('Galaxies - No Data', fontsize=12, pad=20)
            ax_bl.grid(True, alpha=0.3)
    else:
        ax_bl.text(0, 0, 'No galaxies data available', ha='center', va='center', 
                  transform=ax_bl.transData, fontsize=12)
        ax_bl.set_title('Galaxies - No Data', fontsize=12, pad=20)
        ax_bl.grid(True, alpha=0.3)
    
    # Top Right: Young stars (only_stars minus final)
    if len(stages_to_plot) >= 2:
        stage_name, stage_df = stages_to_plot[2]
        if stage_name == 'young_stars' and len(stage_df) > 0:
            create_polar_histogram(ax_tr, stage_df,
                                  f'{STAGE_STYLES[stage_name]["label"]} - {len(stage_df)} sources',
                                  'Source Count')
        else:
            ax_tr.text(0, 0, 'No young stars data available', ha='center', va='center', 
                      transform=ax_tr.transData, fontsize=12)
            ax_tr.set_title('Young Stars - No Data', fontsize=12, pad=20)
            ax_tr.grid(True, alpha=0.3)
    else:
        ax_tr.text(0, 0, 'No young stars data available', ha='center', va='center', 
                  transform=ax_tr.transData, fontsize=12)
        ax_tr.set_title('Young Stars - No Data', fontsize=12, pad=20)
        ax_tr.grid(True, alpha=0.3)
    
    # Bottom Right: Old stars (final)
    if len(stages_to_plot) >= 3:
        stage_name, stage_df = stages_to_plot[3]
        if stage_name == 'final' and len(stage_df) > 0:
            create_polar_histogram(ax_br, stage_df,
                                  f'{STAGE_STYLES[stage_name]["label"]} - {len(stage_df)} sources',
                                  'Source Count')
        else:
            ax_br.text(0, 0, 'No old stars data available', ha='center', va='center', 
                      transform=ax_br.transData, fontsize=12)
            ax_br.set_title('Old Stars - No Data', fontsize=12, pad=20)
            ax_br.grid(True, alpha=0.3)
    else:
        ax_br.text(0, 0, 'No old stars data available', ha='center', va='center', 
                  transform=ax_br.transData, fontsize=12)
        ax_br.set_title('Old Stars - No Data', fontsize=12, pad=20)
        ax_br.grid(True, alpha=0.3)
    
    # Add a descriptive title
    if last_photometric_filter:
        filter_description = f"Photometric filter used for 'only stars': {last_photometric_filter}"
    else:
        filter_description = "Using detected photometric filters"
    
    fig.suptitle(f'{current_galaxy.name}: Polar Density Comparison (NIRCAM Masked)\n'
                f'Filters: {F1} & {F2} | {filter_description}', 
                fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig("comparison_histogram.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_radial_density_profile(filtered_dataframes, r_halflight, min_coverage=1.0, last_photometric_filter=None):
    """Figure 7: Radial density profile masked to NIRCAM footprint."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    df_all = filtered_dataframes.get('ALL', pd.DataFrame())
    
    if len(df_all) == 0:
        print("No unfiltered data for radial profile")
        return
    
    all_radius, _ = cart2pol(df_all['RA'], df_all['DEC'],
                             current_galaxy.RA, current_galaxy.DEC)
    max_r = all_radius.max()
    
    gap_deg = NIRCAM_GAP_ARCSEC / 3600.0
    radial_step = gap_deg / step_factor
    
    n_radial_bins = max(5, int(np.ceil(max_r / radial_step)))
    
    bin_edges = np.linspace(0, max_r, n_radial_bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
    
    try:
        footprint = NIRCAMFootprint(df_all, current_galaxy.RA, current_galaxy.DEC)
    except Exception as e:
        print(f"Could not create footprint: {e}")
        footprint = None
    
    for stage, df in stages_to_plot:
        if len(df) > 0 and stage in STAGE_STYLES and 'RA' in df.columns:
            style = STAGE_STYLES[stage]
            radius, _ = cart2pol(df['RA'], df['DEC'],
                                 current_galaxy.RA, current_galaxy.DEC)
            
            density = np.zeros(len(bin_edges)-1)
            poisson_error = np.zeros(len(bin_edges)-1)
            
            for i in range(len(bin_edges)-1):
                r_min, r_max = bin_edges[i], bin_edges[i+1]
                
                in_bin = (radius >= r_min) & (radius < r_max)
                count = np.sum(in_bin)
                
                if footprint is not None:
                    n_theta = 36
                    n_r = 5
                    thetas = np.linspace(0, 2*np.pi, n_theta)
                    rs = np.linspace(r_min, r_max, n_r)
                    
                    inside_samples = 0
                    total_samples = 0
                    
                    for theta in thetas:
                        for r in rs:
                            ra_sample = current_galaxy.RA + r * np.cos(theta) / np.cos(np.radians(current_galaxy.DEC))
                            dec_sample = current_galaxy.DEC + r * np.sin(theta)
                            
                            if footprint.contains_point(ra_sample, dec_sample):
                                inside_samples += 1
                            total_samples += 1
                    
                    total_annulus_area = np.pi * (r_max**2 - r_min**2)
                    effective_area = total_annulus_area * (inside_samples / max(1, total_samples))
                else:
                    effective_area = np.pi * (r_max**2 - r_min**2)
                
                if effective_area > 0:
                    density[i] = count / effective_area
                    if count > 0:
                        poisson_error[i] = np.sqrt(count) / effective_area
            
            valid_idx = density > 0
            if np.any(valid_idx):
                ax.errorbar(bin_centers[valid_idx], density[valid_idx], 
                           yerr=poisson_error[valid_idx],
                           marker=style['marker'], markersize=3,
                           color=style['color'], alpha=style['alpha'],
                           linestyle='none',
                           capsize=3,
                           label=f"{style['label']} ({len(df)} sources)")
    
    ax.set_xlabel('Radius (degrees)')
    ax.set_ylabel('Surface Density (counts/deg²)')
    ax.set_title(f'{current_galaxy.name}: Radial Density Profile\n'
                f'Filters: {F1} & {F2}')
    ax.set_yscale('log')
    
    def deg_to_kpc(x): 
        return (np.pi * x * current_galaxy.dist)/180
    def kpc_to_deg(x): 
        return (180 * x)/(np.pi * current_galaxy.dist)
    secax = ax.secondary_xaxis('top', functions=(deg_to_kpc, kpc_to_deg))
    secax.set_xlabel('kpc')
    
    if 'final' in filtered_dataframes and len(filtered_dataframes['final']) > 0:
        r = np.linspace(0, max_r, 100)
        y_param = 1.678
        r_exp = r_hl/y_param
        N_tot = len(filtered_dataframes['final'])
        sigma_0 = N_tot/(2*np.pi*r_exp**2)
        sigma = sigma_0 * np.exp(-r/r_exp)
        ax.plot(r, sigma, color='g', linewidth=2, label="Central disk theory", alpha=0.7)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("profile.png", dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
def print_summary_statistics(filtered_dataframes, last_photometric_filter):
    """Print summary statistics for the derived datasets."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Get the three stages
    stages_to_plot = get_stages_to_plot(filtered_dataframes, last_photometric_filter)
    
    for stage, df in stages_to_plot:
        if len(df) > 0:
            print(f"\n{STAGE_STYLES[stage]['label']}:")
            print(f"  Total sources: {len(df)}")
            
            if stage == 'final':
                # For old stars, show magnitude range
                if f'{F1}_mag' in df.columns:
                    mag_min = df[f'{F1}_mag'].min()
                    mag_max = df[f'{F1}_mag'].max()
                    print(f"  {F1} magnitude range: {mag_min:.2f} - {mag_max:.2f}")
            
            elif stage == 'young_stars':
                print("  These are stars that passed photometric filters but NOT the isochrone filter")
                print("  (likely younger stellar populations)")
            
            elif stage == 'galaxies':
                print(f"  These are extended/unresolved sources removed by {last_photometric_filter}")
    
    print("\n" + "="*60)

# ============================================================================
# FILEPATH CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HALOS_DIR = SCRIPT_DIR

THEORY_ISOCHRONES_DIR = os.path.join(HALOS_DIR, "Theory Isochrones")
JWST_DATA_DIR = os.path.join(HALOS_DIR, "JWST_Data")
INFO_CSV_PATH = os.path.join(HALOS_DIR, "JWST_GLX_INFO_with_halflight.csv")
#ISOCHRONE_FILE = os.path.join(THEORY_ISOCHRONES_DIR, "12000JWST_VEGA_11_13.isc_jwst-nircam_PL")
ISOCHRONE_FILE = os.path.join(THEORY_ISOCHRONES_DIR, "12500_JWST_VEGA_LEOA.isc_jwst-nircam_PL")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

df_isochrone = pd.read_csv(ISOCHRONE_FILE, sep='\s+')

data_types = {
    'NAME': str,
    'RA': float,
    'DEC': float,
    'DISTANCE': float,
    'FILTER1': str,
    'FILTER2': str,
    'HALFLIGHT_RAD': float
}
INFO = pd.read_csv(INFO_CSV_PATH, dtype=data_types)

path = JWST_DATA_DIR
glx = np.zeros(len(INFO), dtype=object)
for i in range(len(INFO)):
    glx[i] = GALAXY(INFO.NAME[i], INFO.RA[i], INFO.DEC[i], INFO.DISTANCE[i], INFO.HALFLIGHT_RAD[i])
    print(str(i) + ": " + glx[i].name)

# ============================================================================
# USER CONTROLS
# ============================================================================
#galaxy selection
num = 1
current_galaxy = glx[num]

#keep these in the code so that it is easy to test modifications for analysis
#source matching
match_radius_arcsec = 0.03 #arcsec

#lim mag
limiting_magnitude = 26 #qualitative or add more rigorous filter application

#JWST photometry filters
CI_threshold = 2.0
ellipticity_threshold = 0.3

#footprint masking, 1.0 means 100%
min_footprint_coverage = 1.0

#isochrone controls
shift_i=0.0
width_i=0.3
age_i=12500 #Myr

#half light radius is by default in kpc,
#but for LEOP it is in arcsec, and SCULPTORB in arcmin,
#so convert appropriately into degrees
r_hl = np.degrees(current_galaxy.hlr/current_galaxy.dist)
if num == 0:
    r_hl = current_galaxy.hlr/3600
if num == 7:
    r_hl = current_galaxy.hlr/60
    
#binning parameters
n_theta_bins = 40
step_factor = 4

# ============================================================================
# PROCESS SELECTED GALAXY
# ============================================================================
F1 = INFO.FILTER1[num]
F2 = INFO.FILTER2[num]

print(f"\n{'='*60}")
print(f"Processing galaxy: {current_galaxy.name}")
print(f"Filters: {F1} and {F2}")
print(f"Distance: {current_galaxy.dist} Mpc")
print(f"Location (RA, DEC): {current_galaxy.RA}, {current_galaxy.DEC}")
print(f"Half-light radius: {current_galaxy.hlr} kpc")
print(f"{'='*60}\n")

galaxy_path = os.path.join(path, current_galaxy.name)
print(f"Changing to directory: {galaxy_path}")
if os.path.exists(galaxy_path):
    os.chdir(galaxy_path)
else:
    print(f"Warning: Directory {galaxy_path} does not exist!")

all_filters = load_all_filters()

if all_filters:
    print(f"\nSuccessfully loaded {len(all_filters)} filters: {list(all_filters.keys())}")
    
    matched_data = match_sources_across_filters(all_filters)
    
    app_A_f1, app_A_f2 = app_mag(age_i, df_isochrone[F1], df_isochrone[F2], current_galaxy.dist)
    app_A_f1 = app_A_f1[0:1250]
    app_A_f2 = app_A_f2[0:1250]
    idx_A = app_A_f1 - app_A_f2
    
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    if matched_data is not None and len(matched_data) > 0:
        
        #add an if statement for plotting to check if jwst photometric filters fall before isochrone one as error case
        custom_filter_pipeline = ['ALL', 'resolved', 'is_extended', 'CI', 'ellipticity', 'isochrone']
        
        filtered_dataframes = apply_filter_pipeline(
            matched_data, 
            current_galaxy.dist,
            filter_list=custom_filter_pipeline,
            isochrone_color=idx_A,
            isochrone_mag=app_A_f2,
            shift_amount=shift_i,
            width=width_i
        )
        
        try:
            #last_photometric_filter = 'ellipticity'
            last_photometric_filter = get_last_photometric_filter(custom_filter_pipeline)
            print(f"Last photometric filter in pipeline: {last_photometric_filter}")
            
            # Create derived dataframes
            filtered_dataframes = create_derived_dataframes(filtered_dataframes, last_photometric_filter)
            print_summary_statistics(filtered_dataframes, last_photometric_filter)
            
            # Now plot using the new derived dataframes
            plot_cmd_single(filtered_dataframes, idx_A, app_A_f2, last_photometric_filter)
            #plot_cumulative_magnitude(filtered_dataframes, last_photometric_filter)
            plot_location_cartesian(filtered_dataframes, last_photometric_filter)
            #plot_polar_location(filtered_dataframes, last_photometric_filter)
            plot_polar_histogram_filtered_only(filtered_dataframes, min_footprint_coverage, last_photometric_filter)
            #plot_polar_histogram_comparison(filtered_dataframes, min_footprint_coverage, last_photometric_filter)
            plot_radial_density_profile(filtered_dataframes, min_footprint_coverage, last_photometric_filter)
            
            
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Cannot proceed with plotting without at least one JWST photometric filter.")
    else:
        print("No matched data found or matched data is empty")
else:
    print("Failed to load filter catalogs")
    
#should already be in correct folder corresponding to galaxy...
#plotting CMD, cartesian location, polar histogram, and radial density profile together
#simple to modify to chose a different set to visualize together
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Load your saved images
# (Adjust filenames to whatever you saved them as)
plot_files = [
    'CMD.png',
    'cartesian_location.png', 
    'polar_histogram.png',
    'profile.png'
]

# Flatten axes for easy indexing
ax_flat = axes.flatten()

# Load and display each image
for i, file_path in enumerate(plot_files):
    img = mpimg.imread(file_path)
    ax_flat[i].imshow(img)
    ax_flat[i].axis('off')  # Hide axes
    ax_flat[i].set_title(f'{["Color-Magnitude-Diagram", "Cartesian Location", "Polar Histogram", "Radial Density Profile"][i]}', fontsize=14)

plt.tight_layout()
plt.savefig('combined_figure.png', dpi=300, bbox_inches='tight')
plt.show()
