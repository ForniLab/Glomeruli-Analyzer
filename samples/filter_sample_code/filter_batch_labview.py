# Version 4.0.1

import glob
import os
import os.path
import statistics
import sys
import time
import warnings
import tifffile as tiff
import matplotlib.pyplot as plt
from datetime import timedelta
from PIL import Image
from statistics import mean

import cv2
import numpy as np
import shutil
from multiprocessing import Process, Manager
from setting_batch import *
from os import walk

# Start Measuring Time
start_time = time.monotonic()

files = []
for (dirpath, dirnames, filenames) in walk(image_source_folder):
    files.extend(filenames)
    break

# ----------------------------------
# if __name__ == "__main__":
# Startup settings
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
sys_path = os.path.abspath("../..")
sys.path.append(sys_path)
# ----------------------------------

from utils.contour_utils.contourutil import ContourUtil
warnings.filterwarnings("ignore",
                        message="invalid value encountered in greater_equal")
warnings.filterwarnings("ignore",
                        message="invalid value encountered in less_equal")


def file_extention_check(filepath, filename):
    global ext, image, actual_channels_number
    # Now we can simply use == to check for equality, no need for wildcards.
    if os.path.splitext(filepath)[1] == ".png":
        message_out.append("Source is an PNG!")
        ext = "png"
        # Import file
        image = cv2.imread(filepath)

    elif (os.path.splitext(filepath)[1] == ".tif"
          or os.path.splitext(filepath)[1] == ".tiff"):
        # print ("Source is an TIF!")
        ext = "tif"

        # Import file as dataset and converting its type
        dataset = Image.open(filepath)

        if len(np.shape(dataset)) == 2:
            message_out.append("It's a normal TIF file.")
            h, w = np.shape(dataset)
            tiffarray = np.zeros((h, w, dataset.n_frames))
            for i in range(dataset.n_frames):
                dataset.seek(i)
                tiffarray[:, :, i] = np.array(dataset)

            image_with_alpha = tiffarray.astype(np.uint8)

            # Color Correction process
            """
            Recepie:
            Here are the TIFs. In these files,
            channels 1 (green),
            channel 2 (gray is what we've been calling blue)
            channel 4 (red)
            """
            image_with_alpha_correct_color_order = np.zeros(
                image_with_alpha.shape).astype(np.uint8)

            if (image_with_alpha.shape[2] == 4):
                image_with_alpha_correct_color_order[:, :,
                                                     0] = image_with_alpha[:, :,
                                                                           order_recepie[0]]
                image_with_alpha_correct_color_order[:, :,
                                                     1] = image_with_alpha[:, :,
                                                                           order_recepie[1]]
                image_with_alpha_correct_color_order[:, :,
                                                     2] = image_with_alpha[:, :,
                                                                           order_recepie[2]]
                image_with_alpha_correct_color_order[:, :,
                                                     3] = image_with_alpha[:, :,
                                                                           ch_to_ignore]

                # Recording actual channel number
                actual_channels_number = image_with_alpha_correct_color_order.shape[
                    2]
                # Reducing number of channels to 3
                image = image_with_alpha_correct_color_order[:, :, :3]
            else:
                image_with_alpha_correct_color_order[:, :,
                                                     0] = image_with_alpha[:, :,
                                                                           order_recepie[0]]
                image_with_alpha_correct_color_order[:, :,
                                                     1] = image_with_alpha[:, :,
                                                                           order_recepie[1]]
                image_with_alpha_correct_color_order[:, :,
                                                     2] = image_with_alpha[:, :,
                                                                           order_recepie[2]]

                # Recording actual channel number
                actual_channels_number = image_with_alpha_correct_color_order.shape[
                    2]
                # Reducing number of channels to 3
                image = image_with_alpha_correct_color_order[:, :, :3]

            if transparent_corrector is True:
                img = transparent_corrector(image)
                image = np.array(img)
                image = image[:, :, :3]
            if zero_pixel_reporting is True:
                zero_pixel_count(image)

        elif len(np.shape(dataset)) == 3:
            message_out.append("It's an altered TIF file.")
            dataset = np.asanyarray(dataset)
            image_with_alpha = dataset.astype(np.uint8)
            image_with_alpha_correct_color_order = np.zeros(
                image_with_alpha.shape).astype(np.uint8)

            if (image_with_alpha.shape[0] == 4):
                image_with_alpha_correct_color_order[:, :,
                                                     0] = image_with_alpha[:, :,
                                                                           order_recepie[0]]
                image_with_alpha_correct_color_order[:, :,
                                                     1] = image_with_alpha[:, :,
                                                                           order_recepie[1]]
                image_with_alpha_correct_color_order[:, :,
                                                     2] = image_with_alpha[:, :,
                                                                           order_recepie[2]]
                image_with_alpha_correct_color_order[:, :,
                                                     3] = image_with_alpha[:, :,
                                                                           ch_to_ignore]

                # Recording actual channel number
                actual_channels_number = image_with_alpha_correct_color_order.shape[
                    2]
                # Reducing number of channels to 3
                tif_normal = True
                image = image_with_alpha_correct_color_order[:, :, :3]
            else:
                image_with_alpha_correct_color_order[:, :,
                                                     0] = image_with_alpha[:, :,
                                                                           order_recepie[0]]
                image_with_alpha_correct_color_order[:, :,
                                                     1] = image_with_alpha[:, :,
                                                                           order_recepie[1]]
                image_with_alpha_correct_color_order[:, :,
                                                     2] = image_with_alpha[:, :,
                                                                           order_recepie[2]]
                # Recording actual channel number
                actual_channels_number = image_with_alpha_correct_color_order.shape[
                    2]
                # Reducing number of channels to 3
                tif_normal = True
                image = image_with_alpha_correct_color_order[:, :, :3]

    else:
        message_out.append(
            "Warning! The extension is not neither PNG nor TIF!")

    message_out.append(filename)
    return


# ------------------ Transparent Corrector ---------------------------------
def transparent_corrector(image):

    img = Image.fromarray(image)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img


# ------------------ Zero Pixel counter ------------------------------------
def zero_pixel_count(image):

    count = 0
    img = Image.fromarray(image)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            count += count
    message_out.append("Total pure zero pixels are: " + str(count))

    return


# ------------------ Get Image Dimensions ----------------------------------


def image_information(image):
    # get dimensions of image
    global main_height, main_width, main_channels

    main_dimensions = np.shape(image)

    # height, width, number of channels in image
    main_height = main_dimensions[0]
    main_width = main_dimensions[1]
    main_channels = main_dimensions[2]

    message_out.append("Image Information: --------------------")
    message_out.append("Image Dimension    : " + str(main_dimensions))
    message_out.append("Image Height       : " + str(main_height))
    message_out.append("Image Width        : " + str(main_width))
    message_out.append("Number of Channels : " + str(main_channels))
    if ext == "tif" or ext == "tiff":
        if actual_channels_number > 3:
            message_out.append("Actually channel number was" +
                               str(actual_channels_number) +
                               "but we have igonred one of them!")
    message_out.append("---------------------------------------")

    return


# ---------------------- Writing to File -----------------------------------


def output_generator():
    output_filename = image_file_name + ".txt"
    if os.path.exists('outputs/' + output_filename):
        os.remove('outputs/' + output_filename)  # this deletes the file
    output_txt = open('outputs/' + output_filename, "x")
    output_txt.write("Size\t%R\t%B\t%G\tAvg-R\tAvg-B\tAvg-G\n")
    output_txt.close

    return


# --------------- Clean Output Folder --------------------------------------


def output_cleaner():
    """ filelist = glob.glob(os.path.join(path_output, "*.*"))
    for f in filelist:
        os.remove(f) """
    shutil.rmtree('outputs')
    os.makedirs('outputs')

    return


# --------------- Image duplicator -----------------------------------------


def image_duplicator(image):
    # An image copy to draw all outlines on the original file
    global original_image_for_outline, image_copy, sample

    original_image_for_outline = image.copy()
    image_copy = image.copy()
    sample = overlay_image.copy()

    return


# ------------- Generate Transparent image----------------------------------


def transparent_img_generator(h, w, pthout):
    global overlay_image
    # transparent image should have 4 channel (4th is alpha)
    transparent_img = np.zeros((h, w, 4), dtype=np.uint8)
    if ext == "png":
        cv2.imwrite(os.path.join(pthout, "Overlay.png"), transparent_img)
        overlay_image = cv2.imread(os.path.join(pthout, "Overlay.png"))

    elif ext == "tif":
        tiff.imsave(os.path.join(pthout, "Overlay.tif"), transparent_img)
        overlay_image = tiff.imread(os.path.join(pthout, "Overlay.tif"))

    return


# ------------- Threshold Applier --------------------------------------


def threshold_apply(switch, adpthr, statthr):
    # Apply Threshold on grayscaled image
    if switch is True:
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size, # it reads the setting
            C=-adpthr,
        )
    else:
        ret, threshold = cv2.threshold(gray, statthr, 255, cv2.THRESH_BINARY)

    return threshold


# ------------- Size Filter ---------------------------------------


def size_filter(cnts, minsz, mazsz):
    n = len(cnts)
    cnts_size = []
    filtered_cnts = []

    for i in range(0, n):
        # Applying 1st filter (size filer)
        size = cv2.contourArea(cnts[i])
        if (size > minsz) and (size < mazsz):
            filtered_cnts.append(cnts[i])
            cnts_size.append(size)

    message_out.append("َAfter 1st filter by size:" + str(len(filtered_cnts)))
    return cnts_size, filtered_cnts


# ------------- Ratio Filter --------------------------------------


def ratio_filter(cnts, ctns_size, margin):
    n = len(cnts)
    filtered_cnts = []
    P_to_S_ratio = []

    for i in range(n):
        perimeter = cv2.arcLength(cnts[i], True)
        P_to_S_ratio.append(perimeter / ctns_size[i])

    Max_Ratio = max(P_to_S_ratio)
    Std_Ratio = statistics.stdev(P_to_S_ratio)

    for i in range(n):
        if P_to_S_ratio[i] < (Max_Ratio - margin * Std_Ratio):
            filtered_cnts.append(cnts[i])

    message_out.append("Max Ratio:" + str(max(P_to_S_ratio)))
    message_out.append("Standard Deviation:" +
                       str(statistics.stdev(P_to_S_ratio)))

    message = "\t(" + str(len(cnts) -
                          len(filtered_cnts)) + " has been removed!)"
    message_out.append("َAfter 2st filter by ratio:" +
                       str(len(filtered_cnts)) + message)
    return filtered_cnts, len(filtered_cnts)


# ------------- Ratio Filter --------------------------------------


def ellipticity_filter(cnts, hist_switch, hist_order, ch, crc, cre):
    n = len(cnts)
    # Extent is the ratio of contour area to sorounding circle area
    extent = []
    extent_ellipse = []
    filtered_cnts = []

    for i in range(n):
        size = cv2.contourArea(cnts[i])
        # Calculating "Extent" parameter to find how it is close to a circle
        (x, y), radius = cv2.minEnclosingCircle(cnts[i])
        circle_area = np.pi * radius * radius
        extent.append((size / circle_area))
        # Calculating how it is close to an ellipse
        if len(cnts[i]) > 5:
            ellipse = cv2.fitEllipse(cnts[i])
            axes = ellipse[1]
            minor, major = axes
            extent_ellipse.append(minor / major)
        else:
            extent_ellipse.append(1)

    # Histogram Generator ----------------------------------------
    if hist_switch is True:
        fig1 = plt.figure(1, figsize=[6.4, 4.8], dpi=800)
        fig2 = plt.figure(2, figsize=[6.4, 4.8], dpi=800)
        
        ax = fig1.add_subplot(3, 1, hist_order)
        ax.hist(extent, density=False, bins=num_bins, rwidth=0.7)
        ax.set_title(label="Channel {}".format(ch))
        ax.set_ylabel("Conuts")
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 1
        if hist_order == 3:
            ax.set_xlabel("(contour size)/(circle area) Ratio")
        fig1.tight_layout()

        bx = fig2.add_subplot(3, 1, hist_order)
        bx.hist(extent_ellipse, density=False, bins=num_bins, rwidth=0.7)
        bx.set_title(label="Channel {}".format(ch))
        if hist_order == 3:
            bx.set_xlabel("(minor)/(major) Ratio")
        fig2.tight_layout()
        
        fig1.savefig(os.path.join(path_output, "circular ratio hist.png"))
        fig2.savefig(os.path.join(path_output, "ellipticity ratio.png"))
        plt.close(fig1)
        plt.close(fig2)
    # -----------------------------------------------------------

    for i in range(n):
        if (extent[i] >= crc) and (extent_ellipse[i] >= cre):
            filtered_cnts.append(cnts[i])

    message = "\t(" + str(len(cnts) -
                          len(filtered_cnts)) + " has been removed!)"
    message_out.append("َAfter 3st filter by ellipticity:" +
                       str(len(filtered_cnts)) + message)

    return filtered_cnts


# ------------- Selection Filter --------------------------------------


def selection_filter(cnts, ch, img, pthout, gborders, thinkness):
    filtered_cnts = []
    flt_img = np.zeros(img.shape).astype(np.uint8)
    imgcopy = img.copy()

    contour_label = 0

    # cnts = imutils.grab_contours(cnts)
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Prevent overshooting text
        if cY < 10 and cX < 10:
            cv2.putText(
                imgcopy,
                str(contour_label),
                (cX + 10, cY + 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                1,
            )
        elif cX < 10:
            cv2.putText(
                imgcopy,
                str(contour_label),
                (cX + 10, cY - 5),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                1,
            )
        elif cY < 10:
            cv2.putText(
                imgcopy,
                str(contour_label),
                (cX - 5, cY + 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                1,
            )
        else:
            cv2.putText(
                imgcopy,
                str(contour_label),
                (cX - 5, cY - 5),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                1,
            )

        contour_label += 1

    cv2.imwrite(os.path.join(pthout, "Channel-{}-labled.png".format(ch)),
                imgcopy)
    user_ready = "n"
    while user_ready != "y":
        user_ready = input(
            "Please prepare filter-number-{}.txt and enter 'y': ".format(ch))

    text_file = open("filter-number-{}.txt".format(ch), "r")

    num_to_remover = text_file.read().split(",")
    num_to_remover = [int(i) for i in num_to_remover]

    for j in range(len(cnts)):
        if j not in num_to_remover:
            filtered_cnts.append(cnts[j])
        else:
            message_out.append("Contour #" + str(j) +
                               "in Channel-{} removed.".format(ch))

    flt_img = cv2.drawContours(flt_img, filtered_cnts, -1, gborders, thinkness)

    cv2.imwrite(os.path.join(pthout, "{}-Bordered-removed.png".format(ch)),
                flt_img)

    return filtered_cnts


# ------------- Biggest Contours Drawer ------------------------------------


def most_bigest_contours_drawer(img, cnts, brcolor, fillbg, ptout, ch, n):
    sample = img.copy()

    for i in range(0, n):
        sample = cv2.drawContours(sample, cnts, i, brcolor, fillbg)
    cv2.imwrite(os.path.join(ptout, "Biggest_Contour(s)_{}.png".format(ch)),
                sample)

    return


# ------------- Color Combination Calculator--------------------------------
def report_contours_point(cnts):

    text_file = open("contour_list.txt", "w")  # Write the
    text_file.write("%s" % cnts)
    text_file.close()

    return


# ------------- Color Combination Calculator--------------------------------


def color_combination_calculator(cnts, img):
    contour_util = ContourUtil()

    result_list = [[], [], [], []]
    text = ''

    for i in range(len(cnts)):
        # Generating mask
        masked = contour_util.create_contour_mask(img, cnts[i])

        # Finding extremum points around contour to crop it
        c = cnts[i]
        extLeft_array = tuple(c[c[:, :, 0].argmin()][0])
        extLeft = extLeft_array[0]
        extRight_array = tuple(c[c[:, :, 0].argmax()][0])
        extRight = extRight_array[0]
        extTop_array = tuple(c[c[:, :, 1].argmin()][0])
        extTop = extTop_array[1]
        extBot_array = tuple(c[c[:, :, 1].argmax()][0])
        extBot = extBot_array[1]
        masked = masked[extTop:extBot, extLeft:extRight]

        # c_sum = contour_util.compute_sum(masked)
        c_avg = contour_util.compute_average(masked)

        # Calculating all colors percentage
        total_avg = c_avg[0] + c_avg[1] + c_avg[2]
        Red_perc = round(c_avg[2] / total_avg * 100, 2)
        Blue_perc = round(c_avg[0] / total_avg * 100, 2)
        Green_perc = round(c_avg[1] / total_avg * 100, 2)

        result_list[0].append(cv2.contourArea(cnts[i]))
        result_list[1].append(c_avg[2])
        result_list[2].append(c_avg[0])
        result_list[3].append(c_avg[1])

        # Display color percentage for debug
        """ print("Total Avg :", total_avg)
        print("% Blue :", Blue_perc)
        print("% Green :", Green_perc)
        print("% Red :", Red_perc) """

        if detailed_output:
            text = text + (str(cv2.contourArea(cnts[i])) + "\t" +
                           str(Red_perc) + "\t" + str(Blue_perc) + "\t" +
                           str(Green_perc) + "\t" + str(c_avg[2]) + "\t" +
                           str(c_avg[0]) + "\t" + str(c_avg[1]) + "\n")

    if detailed_output:
        output_txt = open('outputs/' + image_file_name + ".txt", "a")
        output_txt.write(text)
        output_txt.close

    return result_list


# ------------ Generating Transparent Overlay layer image------------------


def transparent_layer_generator(img, ptout):
    color_layer = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(color_layer, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    transparent_overlay = cv2.merge(rgba, 4)
    cv2.imwrite(os.path.join(ptout, "transparent-overlay.png"),
                transparent_overlay)

    return


# -------------------- Void Overlay Remover ------------------------------


def hierarchy_sorter(cnts_untouched, cnts_sorted, hier):

    result = []
    list_length = len(a)
    for i in range(list_length):
        result.append(max(a[i], b[i]))

    message_out.append(result[0])
    sz = []
    for i in range(len(cnts_untouched)):
        sz.append(cv2.contourArea(cnts_untouched[i]))

    newlist = map(list, zip(cnts_untouched, hier))
    inner_contours, new_hierarchy = map(
        *sorted(newlist, key=lambda x: (x[0]), reverse=True))

    # return the list of sorted contours
    message_out.append(inner_contours)
    """ inner_contours = []
    new_hierarchy = []
    
    inner_contours.append(cnts_untouched[1])
    new_hierarchy.append(hier[0][1]) """

    return inner_contours, np.array(new_hierarchy, dtype=np.int32)


# -------------------- Void Overlay Remover ------------------------------


def remove_void_overlay(ext, pthout):
    if ext == "png":
        # Delete generated transparent PNG
        os.remove(os.path.join(pthout, "Overlay.png"))
    elif ext == "tif":
        # Delete generated transparent PNG
        os.remove(os.path.join(pthout, "Overlay.tif"))


# -------------------- Max Pixel Value Finder-----------------------------


def max_pixel_value(img):
    message_out.append("Max Pixel Value :" + str(np.amax(img)))

    return


# -------------------- Averager -----------------------------


def normalized_merger(f_name, lst):
    summerized_result_new = [
        f_name, mean(lst[1]),
        mean(lst[2]),
        mean(lst[3]),
        mean(lst[4])
    ]

    for i in range(1, 4):
        max_val = max(lst[i + 1])
        normalized = summerized_result_new[i + 1] * 100 / max_val
        summerized_result_new.append(normalized)

    return summerized_result_new


# ----------------- Image Processing -------------------------------------
def main(f, final_lst):

    global gray, original_image_for_outline, overlay_image, image_file_name, path_output, message_out, num_total_contours

    summerized_result = [f, [], [], [], []]
    num_total_contours = 0

    message_out = []
    image_file_name = f
    path_output = "outputs/" + f + "/"

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # ----------------------Setting--------------------------------------------
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(image_source_folder, image_file_name)
    # ------------------ File Extension Checker--------------------------------

    file_extention_check(filepath, f)
    image_information(image)

    if detailed_output:
        output_generator()

    transparent_img_generator(main_height, main_width, path_output)
    image_duplicator(image)

    for k in ch.keys():
        if enable_channel_for_analysis[k]:
            # ?
            filtered_img = np.zeros(image.shape).astype(np.uint8)
            filtered_img_removed_contours = np.zeros(
                image.shape).astype(np.uint8)

            # Pick Each Color from Original image
            filtered_img[:, :, ch[k]] = image[:, :, ch[k]]

            max_pixel_value(filtered_img)

            # Generating single color output file
            cv2.imwrite(os.path.join(path_output + "{}-RGB.png".format(k)),
                        filtered_img)

            # Grayscale filter on single color output
            # gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
            gray = filtered_img[:, :, ch[k]]

            # Apply Threshold on grayscaled image
            threshold = threshold_apply(enable_adp_threshold[k],
                                        Adp_Threshold_vals[k], Threshold_vals[k])

            # Generating grayscaled with applied threshold
            cv2.imwrite(os.path.join(path_output, "Threshold-{}.png".format(k)),
                        threshold)

            # Contour detection logic
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            if contours_details_reporting is True:
                report_contours_point(contours)

            # Sort all contours by largest to Smallest Size
            # We would use this sorted contours for the rest of the process
            sorted_contours = sorted(
                contours, key=cv2.contourArea, reverse=True)

            # counting number of detected contours
            message_out.append(
                "Number of contours detected in {} Channel = {}".format(
                    k, len(contours)))

            # Check if threshold is appropriate
            if len(sorted_contours) > 0:

                # 1st Filter
                contour_size, filtered_contours = size_filter(
                    sorted_contours, min_size_limit[k], max_size_limit[k])

                # Protect algorithm to face error when minimum size threshold
                # was too high or we just have 1 or 2 contours
                # which statitical functions doesn't work well
                if len(contour_size) > 2:

                    # Number of accepted contours in size filter process
                    contours_num_1st = len(filtered_contours)

                    # 2nd filter to remove tails (Ratio Filter)
                    if enable_ratio_filter is True:
                        filtered_contours, contours_num_2nd = ratio_filter(
                            filtered_contours, contour_size,
                            Remove_Ratio_Margin_ch[k])

                    # 3rd filter (Ellipse Filter)
                    if enable_ellipse_filter is True:
                        filtered_contours = ellipticity_filter(
                            filtered_contours,
                            enable_hist_generator,
                            hist_subplot_indx[k],
                            k,
                            Critical_ratio_Circle[k],
                            Critical_ratio_Ellipse[k],
                        )

                    message_out.append("Maximum size was:" +
                                       str(cv2.contourArea(sorted_contours[0])))

                    # Draw outlines on signle color image
                    filtered_img = cv2.drawContours(
                        filtered_img,
                        filtered_contours,
                        -1,
                        Genral_Borders_Color,
                        single_clr_border_thikness,
                    )

                    # 4th Filter (Selection Filter)
                    if enable_selection_filter[k] is True:
                        filtered_contours = selection_filter(
                            filtered_contours,
                            k,
                            filtered_img,
                            path_output,
                            Genral_Borders_Color,
                            single_clr_border_thikness,
                        )

                    # Generate Largest Contour in each color
                    if enable_most_biggest_contours_output is True:
                        most_bigest_contours_drawer(
                            filtered_img_removed_contours,
                            filtered_contours,
                            Channel_Border_Color_tr[k],
                            fill_black_bgd,
                            path_output,
                            k,
                            desired_number,
                        )

                    # Result list contains all contours information in the certain channel
                    result_lst = color_combination_calculator(
                        filtered_contours, image_copy)

                    for i in range(1, 5):
                        summerized_result[i].extend(result_lst[i - 1])

                    # Generating single color output with contour outlines
                    cv2.imwrite(
                        os.path.join(path_output, "{}-Bordered.png".format(k)),
                        filtered_img,
                    )

                    # Generating full colored output with contour outlines -----------
                    # Draw outlines on original full colored image
                    original_image_for_outline = cv2.drawContours(
                        original_image_for_outline,
                        filtered_contours,
                        -1,
                        Channel_Border_Color[k],
                        full_clr_border_thikness,
                    )

                    # (Original image with regions)
                    cv2.imwrite(
                        os.path.join(path_output,
                                     "Original-Bordered.png".format(k)),
                        original_image_for_outline,
                    )

                    # Recallingoutput original bordered image for the next iteration
                    original_image_for_outline = cv2.imread(
                        os.path.join(path_output, "Original-Bordered.png"))

                    # Generating Black-Background Overlay layer image--------------------------
                    overlay_image = cv2.drawContours(
                        overlay_image,
                        filtered_contours,
                        -1,
                        Channel_Border_Color_tr[k],
                        fill_black_bgd,
                    )

                    cv2.imwrite(
                        os.path.join(
                            path_output, "BlackBackGround-overlay.png"),
                        overlay_image,
                    )

                    overlay_image = cv2.imread(
                        os.path.join(path_output, "BlackBackGround-overlay.png"))

                    # Generating Transparent Overlay layer image----------------------------
                    transparent_layer_generator(overlay_image, path_output)

                else:
                    message_out.append(
                        "------------------------- Warning --------------------------------"
                    )
                    message_out.append((
                        "Error! The {}-Minimum threshold Size is too high, which leads to find no contour."
                    ).format(k))
                    message_out.append("We have just found " +
                                       str(len(contour_size)) + "contours!")
                    message_out.append(
                        "------------------------- ------- --------------------------------\n"
                    )

            else:
                message_out.append(
                    "------------------------- Warning --------------------------------"
                )
                message_out.append((
                    "Error! The {}-Threshold is too high, which leads to find no contour."
                ).format(k))
                message_out.append(
                    "------------------------- ------- --------------------------------"
                )

            num_total_contours += len(filtered_contours)

    # This variables are subjected to multiprocessing result recording
    newlist = normalized_merger(f, summerized_result)
    newlist.insert(1, (num_total_contours))
    final_lst.append(newlist)

    # ----------------------------------------------------------------------------

    # Generate a combination of overlay image on the original image
    added_image = cv2.addWeighted(overlay_image, overlay_transparency, image,
                                  1 - overlay_transparency, 0)
    cv2.imwrite(os.path.join(path_output, "combined.png"), added_image)

    remove_void_overlay(ext, path_output)

    message_out.append("Successful!\n\n")
    message_out = '\n'.join(message_out)
    print(message_out)
    #cv2.waitKey(0)

    return


def run():
    output_cleaner()
    
    final_result = []
    final_result.append([
            'File name', '# contours', 'Size avg', 'R avg', 'B avg', 'G avg', 'N. R avg',
            'N. B avg', 'N. G avg'
        ])
        
    for f in files:
        main(f,final_result)
        
    
    with open('outputs/reduced_output.txt', 'a') as filehandle:
            for listitem in final_result:
                for element in listitem:
                    filehandle.write('%s\t' % element)

                filehandle.write('\n') 
                
    print(len(files), "files processed.\n")
    # End of Measuring time
    end_time = time.monotonic()
    print("\nDuration:", timedelta(seconds=(end_time - start_time)))

run()
