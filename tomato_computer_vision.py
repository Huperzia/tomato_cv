# Import libraries
import cv2
import numpy as np
from plantcv import plantcv as pcv
import csv
import sys
import gc
import warnings
import os
import pandas as pd
import traceback
from datetime import datetime
from scipy import stats

# run the script with inputs:
# [1] image path
# [2] results directory
# [3] color_card input photo (optional)

# create all the results directories if they don't already exist
# fruit data file directory
if not os.path.exists(f"{sys.argv[2]}/1.fruit_data"):
	os.mkdir(f"{sys.argv[2]}/1.fruit_data")
# proof image directory
if not os.path.exists(f"{sys.argv[2]}/2.proofs"):
	os.mkdir(f"{sys.argv[2]}/2.proofs")
# individual fruit photos directory
if not os.path.exists(f"{sys.argv[2]}/3.fruit_photos"):
	os.mkdir(f"{sys.argv[2]}/3.fruit_photos")
# color card matrices directory
if not os.path.exists(f"{sys.argv[2]}/4.color_matrices"):
	os.mkdir(f"{sys.argv[2]}/4.color_matrices")
# color corrected photos directory
if not os.path.exists(f"{sys.argv[2]}/5.color_corrected_photos"):
	os.mkdir(f"{sys.argv[2]}/5.color_corrected_photos")
# troubleshooting (errors) directory
if not os.path.exists(f"{sys.argv[2]}/6.troubleshooting"):
	os.mkdir(f"{sys.argv[2]}/6.troubleshooting")
        
# create the scaling data file if it doesn't already exist
if not os.path.exists(f"{sys.argv[2]}/scaling_data.csv"):
    pd.DataFrame(columns=["image", "ruler_location", "rectangle_number", "trait", "value"]).to_csv(f"{sys.argv[2]}/scaling_data.csv", index=False)

# create the troubleshooting file if it doesn't already exist:
open(f"{sys.argv[2]}/troubleshooting.txt", "a").close()

# set the options
class options:
    def __init__(self):        
        self.image_path = str(sys.argv[1])
        self.image_name = self.image_path.split("/")[-1].split(".")[0]
        # if a color correction image path was given:
        if len(sys.argv) > 3:
            self.color_card_path = str(sys.argv[3])
        self.scaling_csv_path = f"./{sys.argv[2]}/scaling_data.csv"
        self.fruit_data_directory = f"./{sys.argv[2]}/1.fruit_data"
        self.proof_directory = f"./{sys.argv[2]}/2.proofs"
        self.indiv_fruit_photos_directory = f"./{sys.argv[2]}/3.fruit_photos"
        self.color_card_matrix_directory = f"./{sys.argv[2]}/4.color_matrices"
        self.color_corrected_photos = f"./{sys.argv[2]}/5.color_corrected_photos"
        self.troubleshooting_directory = f"./{sys.argv[2]}/6.troubleshooting"
        self.troubleshooting_file = f"./{sys.argv[2]}/troubleshooting.txt"
        self.debug = "none"
        self.writeimg = False

# Get options
args = options()

# clear the outputs to prevent anything from the previous run from saving
pcv.outputs.clear()

# Set debug to the global parameter
pcv.params.debug = args.debug

# Set plotting size (default = 100)
pcv.params.dpi = 100

# Increase text size and thickness to make labels clearer
pcv.params.text_size = 10
pcv.params.text_thickness = 20


####################################################################
# READ IN AND COLOR CORRECT THE IMAGE

try: 
    # read in the source image
    img, source_path, source_filename = pcv.readimage(filename=args.image_path)

    # find the color card in the source image
    s_df, s_start, s_space = pcv.transform.find_color_card(rgb_img=img, label=f"{args.image_name}_source")

    # filter out the squares in the source image that aren't part of the color checker (based on location)
    s_filtered = s_df[(s_df.x > 4500) & (s_df.y > 600)]

    # find the smallest x value and smallest y value in each dataframe (to account for undetected color chips)
    s_starting_x = s_filtered["x"].min()
    s_starting_y = s_filtered["y"].min()

    # set the size of the circles in the source image
    s_radius = 30
    s_spacing = 263

    # if a color card image was input:
    if hasattr(args, "color_card_path"):
        # read in the color card
        cc_array, cc_path, cc_filename = pcv.readimage(filename=args.color_card_path)

        # resize the target_image
        resized_array = np.zeros((4000, 6000, 3), np.uint8)
        resized_array[0:cc_array.shape[0],0:cc_array.shape[1]] = cc_array

        # find the color card in target image
        t_df, t_start, t_space = pcv.transform.find_color_card(rgb_img=resized_array, label=f"{args.image_name}_target")

        # find the smallest x value and smallest y value in each dataframe
        t_starting_x = t_df["x"].min()
        t_starting_y = t_df["y"].min()

        # create the color card mask for both images
        t_radius = 32
        t_spacing = 288
        s_mask = pcv.transform.create_color_card_mask(rgb_img=img, radius=s_radius, start_coord=(s_starting_x, s_starting_y), spacing=(s_spacing, s_spacing), ncols=4, nrows=6)
        t_mask = pcv.transform.create_color_card_mask(rgb_img=resized_array, radius=t_radius, start_coord=(t_starting_x, t_starting_y), spacing=(t_spacing, t_spacing), ncols=4, nrows=6)

        # run the color correction 
        tm, sm, transformation_matrix, corrected_img = pcv.transform.correct_color(target_img=resized_array, target_mask=t_mask, 
                                                                                source_img=img, source_mask=s_mask, 
                                                                                output_directory=args.color_card_matrix_directory)

        # find and save the color matrix of the source, target, and corrected image
        sc_mask = pcv.transform.create_color_card_mask(rgb_img=corrected_img, radius=s_radius, start_coord=(s_starting_x, s_starting_y), spacing=(s_spacing, s_spacing), ncols=4, nrows=6)
        sc_matrix = pcv.transform.get_color_matrix(rgb_img=corrected_img, mask=sc_mask)[1]
        pcv.transform.save_matrix(matrix=sm, filename=f"{args.color_card_matrix_directory}/{args.image_name}_source.npz")
        pcv.transform.save_matrix(matrix=sc_matrix, filename=f"{args.color_card_matrix_directory}/{args.image_name}_source_corrected.npz")
        pcv.transform.save_matrix(matrix=tm, filename=f"{args.color_card_matrix_directory}/{args.image_name}_target.npz")

        # save the corrected image
        cv2.imwrite(filename=f"{args.color_corrected_photos}/{args.image_name}.jpg", img=corrected_img)

        # reassign the corrected img to the original img path
#        img = corrected_img          

    # clear the outputs
    pcv.outputs.clear()

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with color correction: {traceback.format_exc()}\n")

####################################################################
# PROCESS THE IMAGE

try:
    # crop the image to the desired region
    x_crop = 700
    y_crop = 700
    h_crop = 3200
    w_crop = 3900
    img_cropped = pcv.crop(img=img, x=x_crop, y=y_crop, h=h_crop, w=w_crop)

    # visualize colorspaces to select a color channel that maximizes the difference between the fruit and the background
    #colorspace_img = pcv.visualize.colorspaces(rgb_img=img_cropped, original_img = False)

    # function that takes in a photo and outputs the thresholded & cleaned photo
    def threshold(img, channel: str, threshold_value: float, noise_size: float, light_or_dark: str, debug: str):
        """Inputs:
            img = the image to be thresholded
            channel = the channel to threshold on ('l', 'a', 'b', 'h', 's', 'v')
            threshold_value = the value to threshold the image
            noise_size = the largest size background noise to filter out
            light_or_dark = whether the objects of interest are lighter or darker than the background ('light' or 'dark')
            debug = debugging mode ('print', 'plot', 'none')
        """
        # set the debugging mode
        pcv.params.debug = debug

        # convert the image to grayscale
        if channel in ["l", "a", "b"]:
            gray_img = pcv.rgb2gray_lab(rgb_img=img, channel=channel)
        else:
            gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel=channel)

        # visualize the distribution of greyscale values using a histogram
        #hist = pcv.visualize.histogram(img=gray_img, title = channel)

        # threshold the greyscale image
        thresh_img = pcv.threshold.binary(gray_img=gray_img, threshold=threshold_value, max_value=255, object_type=light_or_dark)

        # remove small background noise
        fill_img = pcv.fill(bin_img=thresh_img, size=noise_size)

        # change the debugging mode back
        pcv.params.debug = args.debug

        # return the cleaned image
        return fill_img

    # threshold and clean the input image
    img_cleaned = threshold(img_cropped, "b", 140, 8000, "light", "none")

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with image processing: {traceback.format_exc()}\n")



####################################################################
# FIND AND READ THE QR CODE

try:
# create a copy of the original image
    proof = np.copy(img)
    proof_qr = np.copy(img)
    proof_cc = np.copy(img)
    proof_sf = np.copy(img)

    # find and read the QR code
    qcd = cv2.QRCodeDetector()
    # use otsu thresholding to find the optimal thresholding value
    qr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, qr_th = cv2.threshold(qr_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(qr_th)

    if retval:
        # save the decoded info to the outputs class
        pcv.outputs.add_observation(sample=args.image_name, variable="accession", trait="accession", \
                                    method="QRcode", scale="none", datatype=str, value=decoded_info[0][0:-1], label="none")

        # plot a rectangle around the QR code on the proof image
        points = points.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img=proof, pts=[points], isClosed=True, color=(80,250,123), thickness=12)
#        cv2.polylines(img=proof_qr, pts=[points], isClosed=True, color=(80,250,123), thickness=12)
#        cv2.fillPoly(img=proof, pts=[points], color=(80,250,123))
#        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    else:
        # save an error message to the troubleshooting file and a photo to the troubleshooting directory
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(args.troubleshooting_file, "a") as f:
            f.write(f"[{time}] [{args.image_name}] QR code not found\n")
        cv2.imwrite(filename=f"{args.troubleshooting_directory}/{args.image_name}_QR.jpg", img=proof)

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with QR code processing: {traceback.format_exc()}\n")


try:
    # save the proof image
    cv2.imwrite(f"{args.proof_directory}/{args.image_name}_proof_qr.jpg", proof)
except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with saving proof image: {traceback.format_exc()}\n")

####################################################################
# CREATE A PROOF IMAGE

# Original input image with:
#   fruit shaded and labeled
#   ruler rectangles shaded
#   color checker outlined

try:
    # if a color card image was given, draw circles in 
    if hasattr(args, "color_card_path"):
        circle_color = (2,67,217)
    else:
        circle_color = (255,184,108)
#        circle_color = (181,74,147)

    # for every row in the color checker
    for i in range(0, 4):
        # for every column in the color checker
        for j in range(0, 6):
            # draw a circle with the same radius as was used in the color correction
            cv2.circle(img=proof, center=(s_starting_x + i*s_spacing, s_starting_y + j*s_spacing), radius=s_radius, color=circle_color, thickness=12)
#            cv2.circle(img=proof_cc, center=(s_starting_x + i*s_spacing, s_starting_y + j*s_spacing), radius=s_radius, color=circle_color, thickness=6)

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with proof image: {traceback.format_exc()}\n")



try:
    # save the proof image
    proof_cc = np.copy(proof)
    cv2.imwrite(f"{args.proof_directory}/{args.image_name}_proof_cc.jpg", proof)
except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with saving proof image: {traceback.format_exc()}\n")



####################################################################
# MAKE SHAPE AND COLOR MEASUREMENTS AND SAVE THE DATA

# to measure the shortest diameter of the fruit: rotate the fruit and measure the height at every angle, save the shortest height
# also measure the longest diameter
# credit: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
# functions to help convert between angles
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


try:
    # identify and count fruit
    fruit_objects_count = cv2.findContours(img_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_fruit = len(fruit_objects_count[1][0])

    # add the number of fruit to the output class
    pcv.outputs.add_observation(sample=args.image_name, variable="num_fruit", trait="num_fruit", \
                                method="findContours", scale="objects", datatype=int, value=num_fruit, label="objects")

    # identify fruit to measure shape and color
    fruit_objects, fruit_hierarchy = pcv.find_objects(img=img_cropped, mask=img_cleaned)

    # function to reorder contours, based off of the specified point (leftmost, topmost, etc.)
    # output the reordered objects and hierarchy list
    def reorder_contours(contours, contour_hierarchy, criteria: str, ascending: bool):
        """Inputs:
        contours = the list of contours
        countour_hierarchy = the object hierarchy
        criteria = how to sort the contours ('leftmost', 'rightmotst', 'topmost', 'bottommost')
        ascending = if true, the objects are output in the ascending order
        """

        # reordered contours list and hierarcy array
        reordered_objects = []
        reordered_hierarchy = [[]]

        # create a dictionary to store the index of the object in the original list : the extreme (ex: leftmost) coordinate of the object
        # iterate through the objects, find the extreme point, and add it to the dictionary
        extreme_dict = {}
        for i in range(len(contours)):
            if criteria == "leftmost":
                extreme = tuple(contours[i][contours[i][:,:,0].argmin()][0])[0]
            elif criteria == "topmost":
                extreme = tuple(contours[i][contours[i][:,:,1].argmin()][0])[1]
            elif criteria == "rightmost":
                extreme = tuple(contours[i][contours[i][:,:,0].argmax()][0])[0]
            elif criteria == "bottommost":
                extreme = tuple(contours[i][contours[i][:,:,1].argmax()][0])[1]
            extreme_dict[i] = extreme

        # add the objects to the new list in order
        while len(extreme_dict) > 0:
            # get the key with the smallest extreme point in the dictionary
            min_extreme_key = min(extreme_dict, key=extreme_dict.get)
            # store the corresponding object and hierarchy in the ordered lists
            reordered_objects.append(contours[min_extreme_key])
            reordered_hierarchy[0].append(contour_hierarchy[0][min_extreme_key])
            
            # remove the key from the dictionary
            extreme_dict.pop(min_extreme_key)

        # if the input argument ascending is false, reverse the order of the lists
        if ascending == False:
            # reverse the order of the objects and hierarchy lists
            reordered_objects.reverse()
            reordered_hierarchy[0].reverse()

        # if the input criteria is not one of the accepted strings, throw an error
        if criteria not in ["leftmost", "topmost", "rightmost", "bottommost"]:
            warnings.warn("Illegal input criteria for object reordering. Criteria must be 'leftmost', 'rightmost', 'topmost', or 'bottommost'.")
            
        # return the reordered objects and hierarchy
        return reordered_objects, reordered_hierarchy


    # create a copy of the RGB image for shape analysis annotations
    shape_img = np.copy(img_cropped)

    # turn off plot debugging
    pcv.params.debug = None

    # set up a counter to count the actual fruit (because some will be filtered out)
    fruit_counter = 0

    # reorder the contours based on leftmost position
    reordered_fruit_objects, reordered_fruit_hierarchy = reorder_contours(fruit_objects, fruit_hierarchy, "leftmost", True)

    # for the proof image:
    # make a copy of the original image as an overlay
    overlay = np.copy(proof_cc)
#    img = np.zeros((100,100,3), dtype=np.uint8)
    # add a filled mask to the fruit
    cv2.drawContours(overlay, reordered_fruit_objects, -1, color=(219, 83, 15), thickness=cv2.FILLED, offset=(x_crop, y_crop))
#    cv2.drawContours(overlay, reordered_fruit_objects, -1, color=(219, 83, 15), thickness=cv2.FILLED, offset=(x_crop, y_crop))
    # make the overlay more transparent and apply it to the proof (alpha=1 is opaque, alpha=0 is transparent)
    cv2.addWeighted(overlay, 0.45, proof, 1 - 0.45, 0, proof)
    # draw the outline of the fruit
    cv2.drawContours(proof, reordered_fruit_objects, -1, color=(252, 86, 3), thickness=2, offset=(x_crop, y_crop))

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with fruit identification: {traceback.format_exc()}\n")



try:
    # measure each fruit by iterating over the objects, identifying the fruit, and extracting shape and color measurements
    for i in range(0, len(reordered_fruit_objects)):
        # Check to see if the object has an offshoot in the hierarchy
        if reordered_fruit_hierarchy[0][i][3] == -1:
            fruit_counter += 1

            # Create an object and a mask for one object
            fruit, fruit_mask = pcv.object_composition(img=img_cropped, contours=[reordered_fruit_objects[i]], hierarchy=np.array([[reordered_fruit_hierarchy[0][i]]]))
            
            # Analyze shape of each fruit
            shape_img = pcv.analyze_object(img=shape_img, obj=fruit, mask=fruit_mask, label=f"{args.image_name}_fruit{fruit_counter}")
            
#            cv2.drawContours(proof, reordered_fruit_objects[i], -1, color=(219, 83, 15), thickness=cv2.FILLED, offset=(x_crop, y_crop))
            # make the overlay more transparent and apply it to the proof (alpha=1 is opaque, alpha=0 is transparent)
#            cv2.addWeighted(overlay, 0.45, proof, 1 - 0.45, 0, proof)
            # draw the outline of the fruit
#            cv2.drawContours(proof, reordered_fruit_objects[i], -1, color=(252, 86, 3), thickness=2, offset=(x_crop, y_crop))

            # print a label on the fruit
            com_x = int(pcv.outputs.observations[f"{args.image_name}_fruit{fruit_counter}"]["center_of_mass"]["value"][0])
            com_y = int(pcv.outputs.observations[f"{args.image_name}_fruit{fruit_counter}"]["center_of_mass"]["value"][1])
            center_of_mass = tuple([com_x, com_y])
            cv2.putText(img=proof, text=f"{fruit_counter}", org=(center_of_mass[0] + x_crop + 10, center_of_mass[1] + y_crop - 10), \
                        fontFace=2, fontScale=2, color=(255,255,255), thickness=2)
            
            # print a circle on the center of the fruit
            cv2.circle(img=proof, center=(center_of_mass[0] + x_crop, center_of_mass[1] + y_crop), radius=12, color=(255,255,255), thickness=-1)
            
            # calculate the smallest height of each fruit using 2 different methods
                # method 1 (smallest_height): rotate the fruit and measure the height at each rotation angle
                # method 2 (shortest_path): rotate the fruit and measure the smallest distance from the center to the edge at each rotation angle
            # set an upper limit for the smallest/largest height and shortest/longest path
            smallest_height = float('inf')
            shortest_path = float('inf')
            longest_path = 0
            largest_height = 0

            # for every rotation angle, measure the height of the fruit
            for angle in range(0, 365):

                # rotate the fruit
                rotated_fruit = rotate_contour(reordered_fruit_objects[i], angle)

                # method 2:
                # find the coordinates with the same x value as the center
                # find the coordinate with the y value that is closest to the center, both above and below
                closest_above = float('inf')
                closest_below = float('inf')
                for coord in rotated_fruit:
                    if coord[0][0] == center_of_mass[0]:
                        if (coord[0][1] < center_of_mass[1]) and (center_of_mass[1] - coord[0][1] < closest_above):
                            closest_above = center_of_mass[1] - coord[0][1]
                        elif (coord[0][1] > center_of_mass[1]) and (coord[0][1] - center_of_mass[1] < closest_below):
                            closest_below = coord[0][1] - center_of_mass[1]
                
                # find the distance and update the shortest_path and longest_path
                path = closest_above + closest_below
                if path < shortest_path:
                    shortest_path = path
                if path > longest_path:
                    longest_path = path
                
                # method 1:
                # get the top and bottom points and update the smallest height
                fruit_topmost = tuple(rotated_fruit[rotated_fruit[:,:,1].argmin()][0])
                fruit_bottommost = tuple(rotated_fruit[rotated_fruit[:,:,1].argmax()][0])
                height = fruit_bottommost[1] - fruit_topmost[1]
                if height < smallest_height:
                    smallest_height = height
                if height > largest_height:
                    largest_height = height
            
            # save the smallest/largest height and shortest/longest path in the outputs class
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="smallest_height", trait="smallest_height", \
                                        method="rotation_method1", datatype=float, value=float(smallest_height), label="pixels", scale="pixels")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="shortest_path", trait="shortest_path", \
                                        method="rotation_method2", datatype=float, value=float(shortest_path), label="pixels", scale="pixels")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="largest_height", trait="largest_height", \
                                        method="rotation_method1", datatype=float, value=float(largest_height), label="pixels", scale="pixels")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="longest_path", trait="longest_path", \
                                        method="rotation_method2", datatype=float, value=float(longest_path), label="pixels", scale="pixels")
            
            # save the image of just the fruit to a new directory
            # crop the fruit mask and the original cropped image to the size of the fruit
            fruit_leftmost = tuple(rotated_fruit[rotated_fruit[:,:,0].argmin()][0])
            fruit_height = pcv.outputs.observations[f"{args.image_name}_fruit{fruit_counter}"]["height"]["value"]
            fruit_width = pcv.outputs.observations[f"{args.image_name}_fruit{fruit_counter}"]["width"]["value"]
            fruit_mask_cropped = pcv.crop(img=fruit_mask, x=fruit_leftmost[0], y=fruit_topmost[1], h=fruit_height, w=fruit_width)
            orig_img_cropped = pcv.crop(img=img_cropped, x=fruit_leftmost[0], y=fruit_topmost[1], h=fruit_height, w=fruit_width)
            
            # split the original cropped image into channels
            b, g, r = cv2.split(orig_img_cropped)

            # merge the alpha channel
            rgba = [b, g, r, fruit_mask_cropped]
            rgba_as_array = cv2.merge(rgba,4)
            
            cv2.imwrite(filename=f"{args.indiv_fruit_photos_directory}/{args.image_name}_fruit{fruit_counter}.png", img=rgba_as_array)
            

            # Analyze color of each fruit  
            #color_img = pcv.analyze_color(rgb_img=img_cropped, mask=fruit_mask, hist_plot_type=None, label=f"{args.image_name}_fruit{fruit_counter}")
            # find the average r, g, b, values of an image (square the values)
            # credit for the squaring: https://sighack.com/post/averaging-rgb-colors-the-right-way#:~:text=The%20typical%20approach%20to%20averaging,color%2C%20sum%20their%20squares%20instead.
            # filter the array to just the non-transparent pixels
            rgba_pixels = rgba_as_array[rgba_as_array[:, :, 3] != 0].astype(np.uint16)
            # calculate the number of pixels
            num_pixels = rgba_pixels.shape[0]
            # extract the average and mode r, g, and b values
            avg_r = np.sqrt(np.sum(np.square(rgba_pixels[:, 2])) / num_pixels)
            avg_g = np.sqrt(np.sum(np.square(rgba_pixels[:, 1])) / num_pixels)
            avg_b = np.sqrt(np.sum(np.square(rgba_pixels[:, 0])) / num_pixels)
            mode = stats.mode(rgba_pixels, keepdims=True)[0][0]
            mode_r = mode[2]
            mode_g = mode[1]
            mode_b = mode[0]

            # write those values to the outputs file
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="avg_r", trait="avg_r", \
                                        method="squared", datatype=float, value=float(avg_r), label="none", scale="none")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="avg_g", trait="avg_g", \
                                        method="squared", datatype=float, value=float(avg_g), label="none", scale="none")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="avg_b", trait="avg_b", \
                                        method="squared", datatype=float, value=float(avg_b), label="none", scale="none")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="mode_r", trait="mode_r", \
                                        method="smallest mode", datatype=float, value=float(mode_r), label="none", scale="none")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="mode_g", trait="mode_g", \
                                        method="smallest mode", datatype=float, value=float(mode_g), label="none", scale="none")
            pcv.outputs.add_observation(sample=f"{args.image_name}_fruit{fruit_counter}", variable="mode_b", trait="mode_b", \
                                        method="smallest mode", datatype=float, value=float(mode_b), label="none", scale="none")


    # Plot the shape analysis
    #pcv.plot_image(img=shape_img)

    # save the output data in a csv
    pcv.outputs.save_results(filename=args.fruit_data_directory + "/" + args.image_name + "_fruit_analysis.csv", outformat="csv")

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with fruit measurements: {traceback.format_exc()}\n")


####################################################################
# MAKE SCALING MEASURMENTS AND SAVE THE DATA

try:
    # set debugging mode
    #pcv.params.debug = "plot"

    # save a dictionary with the cropped x and y offsets
    ruler_offsets = {
        "side": [150, 1600],
        "top_l": [1600, 170],
        "top_r": [3150, 170]
    }

    # crop the photo to the regions with the rulers
    ruler_side = pcv.crop(img=img, x=ruler_offsets["side"][0], y=ruler_offsets["side"][1], h=1200, w=500)
    ruler_top_l = pcv.crop(img=img, x=ruler_offsets["top_l"][0], y=ruler_offsets["top_l"][1], h=520, w=1300)
    ruler_top_r = pcv.crop(img=img, x=ruler_offsets["top_r"][0], y=ruler_offsets["top_r"][1], h=520, w=1300)

    # turn off debugging more
    pcv.params.debug = "none"

    # add the cropped photos to a dictionary
    ruler_photos = {
        "side": ruler_side,
        "top_l": ruler_top_l,
        "top_r": ruler_top_r
        }

    # open the scaling data csv in append mode and create the writer
    scaling_csv = open(args.scaling_csv_path, "a", newline="")
    scaling_writer = csv.writer(scaling_csv)

    # function to find the rectangles and save the data about each rectangle to the scaling file
    # also draw the rectangles on the proof image
    def save_rect_data(cropped_photo, cleaned_photo, proof_photo, region: str, debug: str):
        """Inputs:
        cropped_photo = the original RGB image, cropped to the ruler region
        cleaned_photo = the cropped photo/mask after thresholding and cleaning
        proof_photo = the proof image (with fruit annotations)
        region = the region the ruler is from ('side', 'top_l', 'top_r')
        debug = debugging mode ('print', 'plot', 'none')
        """
        # set the debugging mode
        pcv.params.debug = debug

        # find contours
        rect_contours, rect_hier = pcv.find_objects(img=cropped_photo, mask=cleaned_photo)

        # reorder the rectangles
        if (region == "top_l") or (region == "top_r"):
            rect_contours, rect_hier = reorder_contours(rect_contours, rect_hier, "leftmost", True)
        elif region == "side":
            rect_contours, rect_hier = reorder_contours(rect_contours, rect_hier, "topmost", False)

        # set up a counter to count the rectangles (because some will be filtered out)
        rect_counter = 0

        # create a copy of the RGB image for shape analysis annotations
        shape_img_ruler = np.copy(cropped_photo)

        for i in range(0, len(rect_contours)):
            # Check to see if the object has an offshoot in the hierarchy
            if rect_hier[0][i][3] == -1:

                # find the minimum area rectangle (can be rotated) - contains all the points with the smallest area
                # outputs: (center(x, y), (width, height), angle of rotation)
                minrect = cv2.minAreaRect(rect_contours[i])

                # find the area of the contour
                rect_area = cv2.contourArea(rect_contours[i])

                # if the height and width are both > 0
                if (minrect[1][0] > 0) and (minrect[1][1] > 0):
                    # if the aspect ratio is around 2 and the area is around what we expect:
                    if ((abs(((minrect[1][0] / minrect[1][1]) - 2)) < 0.2) or (abs(((minrect[1][1] / minrect[1][0]) - 2)) < 0.2)) and \
                    (10000 < rect_area < 20000):
                        rect_counter += 1

                        # save the width and height
                        if minrect[1][0] >= minrect[1][1]:
                            rect_width = minrect[1][0]
                            rect_height = minrect[1][1]
                        else:
                            rect_width = minrect[1][1]
                            rect_height = minrect[1][0]

                        # write the information to the scaling data file
                        # image, ruler_location, rectangle_number, trait, value
                        scaling_writer.writerows([
                            [args.image_name, region, rect_counter, "area", rect_area], 
                            [args.image_name, region, rect_counter, "width", rect_width], 
                            [args.image_name, region, rect_counter, "height", rect_height]])
                        
                        # for the proof image:
                        # draw the outline of the rectangle
                        cv2.drawContours(proof_photo, rect_contours, i, (50, 7, 242), thickness=1, \
                                                        offset=(ruler_offsets[region]))
                        # draw the filled rectangle on the proof image
                        cv2.drawContours(proof_photo, rect_contours, i, (153, 136, 227), thickness=cv2.FILLED, \
                                        offset=(ruler_offsets[region]))
                        # label the rectangle with the rectangle number
                        com_x = int(minrect[0][0])
                        com_y = int(minrect[0][1])
                        center_of_mass = tuple([com_x, com_y])
                        cv2.putText(img=proof_photo, text=f"{rect_counter}", \
                                    org=(center_of_mass[0] + ruler_offsets[region][0] - 15, \
                                        center_of_mass[1] + ruler_offsets[region][1] + 15), \
                                    fontFace=2, fontScale=2, color=(0, 0, 0), thickness=2)
                        
        # save an error image and text if there are fewer/more rectangles than expected
        if (region == "side" and rect_counter != 3) or (region != "side" and rect_counter != 5):
            cv2.imwrite(filename=f"{args.troubleshooting_directory}/{args.image_name}_{region}.jpg", img=shape_img_ruler)
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(args.troubleshooting_file, "a") as f:
                f.write(f"[{time}] [{args.image_name}] Error with ruler {region}: found {rect_counter} rectangles\n")

        
        # plot the ruler shape analysis image
        #pcv.plot_image(img=shape_img_ruler)

    # for each ruler region
    for region in ruler_photos:

        # threshold the cropped image using otsu thresholding
        ruler_gray = cv2.cvtColor(ruler_photos[region], cv2.COLOR_BGR2GRAY)
        ret, ruler_th = cv2.threshold(ruler_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # find the rectangles and save the data
#        save_rect_data(ruler_photos[region], ruler_th, proof, region, "none")
        save_rect_data(ruler_photos[region], ruler_th, proof_cc, region, "none")
        save_rect_data(ruler_photos[region], ruler_th, proof, region, "none")

    # close the file after writing to it
    scaling_csv.close()

except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with scaling data: {traceback.format_exc()}\n")



try:
    # save the proof image
    cv2.imwrite(f"{args.proof_directory}/{args.image_name}_proof_sf.jpg", proof_cc)
except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with saving proof image: {traceback.format_exc()}\n")


try:
    # save the proof image
    cv2.imwrite(f"{args.proof_directory}/{args.image_name}_proof.jpg", proof)
except Exception as e:
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.troubleshooting_file, "a") as f:
        f.write(f"[{time}] [{args.image_name}] Error with saving proof image: {traceback.format_exc()}\n")

# clear memory (will print the number of unreachable objects found)
gc.collect()
