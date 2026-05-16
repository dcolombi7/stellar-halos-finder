HALOS: A Tool for Filtering and Visualizing Stellar Halo Radial Density Profiles from JWST NIRCam Observations

Author: Daniela Sofia Colombi | University of Maryland - College Park | May 15, 2026

PURPOSE:
Process JWST NIRCAM observations to identify and analyze stellar halos in nearby dwarf galaxies using extendedness, concentration index, ellipticity, and theoretical isochrone matching.

REPOSITORY STRUCTURE:
├── WORKING_JWST.py              # Main analysis pipeline
├── Theory Isochrones/           # BaSTI isochrone files
│   └── 12500_JWST_VEGA_LEOA.isc_jwst-nircam_PL
├── JWST_Data/                   # Galaxy-specific JWST catalogs
│   └── [Galaxy Name]/           # Per-galaxy directories
│       └── *cat.ecsv            # JWST ECSV catalog files
└── JWST_GLX_INFO_with_halflight.csv  # Galaxy parameters

REQUIRED INPUT FILES:

1. Galaxy Information CSV (JWST_GLX_INFO_with_halflight.csv) columns:
   - NAME: Galaxy name (matches folder in JWST_Data)
   - RA: Right Ascension (degrees)
   - DEC: Declination (degrees)
   - DISTANCE: Distance (Mpc)
   - FILTER1: First JWST filter (e.g., 'F090W')
   - FILTER2: Second JWST filter (e.g., 'F150W')
   - HALFLIGHT_RAD: Half-light radius (kpc)

2. JWST ECSV Catalogs (place in JWST_Data/[Galaxy Name]/):
   - Filenames must contain filter name (e.g., '*F090W*cat.ecsv')
   - Required columns: sky_centroid.ra, sky_centroid.dec, aper50_vegamag, is_extended, CI_50_30, ellipticity

3. Theoretical Isochrones: BaSTI files with JWST NIRCAM magnitudes in VEGAMAG system

DEPENDENCIES:
pip install pandas matplotlib numpy astropy scipy shapely scikit-learn

USER-CONFIGURABLE PARAMETERS:
num = 1                              # Galaxy index to process
match_radius_arcsec = 0.03           # Source matching radius (arcsec)
limiting_magnitude = 26              # Magnitude cutoff
CI_threshold = 2.0                   # Concentration Index threshold
ellipticity_threshold = 0.3          # Ellipticity threshold
shift_i = 0.0                        # Isochrone color shift
width_i = 0.3                        # Isochrone matching width
age_i = 12500                        # Isochrone age (Myr)
n_theta_bins = 40                    # Number of polar angle bins
min_footprint_coverage = 1.0         # Minimum bin coverage fraction
step_factor = 4                      # Radial binning factor

FILTER PIPELINE (applied in order):
1. ALL - No filtering, baseline catalog
2. resolved - Placeholder for resolved sources
3. is_extended - Selects point sources (is_extended == 0)
4. CI - Filters by Concentration Index (< threshold)
5. ellipticity - Filters by ellipticity (< threshold)
6. isochrone - Matches theoretical isochrone with color shift

OUTPUT FILES:
CMD.png                    - Color-Magnitude Diagram
cartesian_location.png     - RA/DEC spatial distribution with NIRCAM footprint
polar_histogram.png        - Polar density histogram (old stars)
profile.png                - Radial density profile with exponential fit
combined_figure.png        - 2x2 composite of all plots

DERIVED DATASETS:
galaxies    - Extended sources (ALL minus photometric stars)
young_stars - Stars passing photometric but not isochrone filters
final       - Old stars passing all filters

NIRCAM FOOTPRINT DETECTION:
- Uses k-means clustering of source positions
- Known detector geometry (132" x 132" detectors, 44" gap)
- Calculates position angle automatically
- Supports optional manual offset adjustment

USAGE:
1. Set up directory structure with isochrones and JWST data
2. Update INFO_CSV_PATH, ISOCHRONE_FILE, and JWST_DATA_DIR paths in WORKING_JWST.py
3. Configure user parameters in the "USER CONTROLS" section
4. Run: python WORKING_JWST.py

SPECIAL CASES:
- Galaxy num=0 (LEOP): half-light radius in arcseconds
- Galaxy num=7 (SCULPTORB): half-light radius in arcminutes

NOTES:
- The pipeline changes working directory to each galaxy's folder
- Footprint masking uses Monte Carlo sampling for coverage fraction
- The isochrone filter requires interpolation of theoretical models

CITATION:
Colombi, D.S. (2026). HALOS: A Tool for Filtering and Visualizing Stellar Halo Radial Density Profiles from JWST NIRCam Observations. University of Maryland.

LICENSE:
For academic and research use. Contact author for permissions.
