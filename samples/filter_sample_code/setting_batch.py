image_source_folder = "images/"

# Histogram setting
# Activate histogram generator 0 or 1
enable_hist_generator = False

hist_subplot_indx = {
    # You can change the order of plots
    "B": 1,
    "G": 2,
    "R": 3,
}

num_bins = 10

# Outline color for single channel outputs
Genral_Borders_Color = (255, 255, 255)

# Borders Thikness
single_clr_border_thikness = 2
full_clr_border_thikness = 1

# Fill inside contours
# In order to fill put -1 otherwise 1 for outlines
fill_black_bgd = -1  # Overlay

# Contours size limits
min_size_limit = {
    "B": 40,
    "G": 40,
    "R": 40,
}
max_size_limit = {
    "B": 30000,
    "G": 30000,
    "R": 30000,
}
# Channel color codes
ch = {
    "B": 0,
    "G": 1,
    "R": 2,
}
# These are static threshold and when adaptive filter is
# off these thresholds apply automatically
Threshold_vals = {
    "B": 20,
    "G": 20,
    "R": 20,
}
Channel_Border_Color = {
    "B": (255, 255, 255),
    "G": (188, 143, 143),
    "R": (255, 255, 0),
}

Channel_Border_Color_tr = {
    "B": (255, 0, 0, 0),
    "G": (0, 255, 0, 0),
    "R": (0, 0, 255, 0),
}

overlay_transparency = 1.000000  # Is used for combined image

# Generating png of interested contours
enable_most_biggest_contours_output = False
desired_number = 3

enable_adp_threshold = {
    "B": True,
    "G": True,
    "R": True,
}
Adp_Threshold_vals = {
    "B": 4,
    "G": 4,
    "R": 2,
}

# Filters
enable_ratio_filter = False
# It filters all contours which have perimeter to area ration greater than:
# max-(*)std
Remove_Ratio_Margin_ch = {
    "B": 1.500000,
    "G": 1.500000,
    "R": 1.500000,
}

enable_ellipse_filter = False

# Should be less than 1, close to one means circle is filled more
# Just pass countors with higher values
Critical_ratio_Circle = {
    # In order to deactive just put zero
    "B": 0.030000,
    "G": 0.000000,
    "R": 0.000000,
}

# Should be less than 1, close to one means close to circle shape
# Just pass countors with higher values
Critical_ratio_Ellipse = {
    # In order to deactive just put zero
    "B":  0.030000,
    "G":  0.000000,
    "R":  0.000000,
}

# Contour remover by number filter
enable_selection_filter = {
    "B": False,
    "G": False,
    "R": False,
}

# channel process
transparent_corrector = True
contours_details_reporting = False
zero_pixel_reporting = False
detailed_output = False

enable_channel_for_analysis = {
    "B": False,
    "G": False,
    "R": True,
}

# adaptive threshold
block_size = 101

# Channel color recepie
ch_to_ignore = 3
order_recepie = [1, 0, 3]