import cv2
import argparse
import os
import numpy as np
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_PNG', type=str, required=True, 
                    help='Path to the directory containing PNG files.')
parser.add_argument('--path_to_saved_ROI', type=str, required=True, 
                    help='Path to the directory to save the extracted ROI images.')
parser.add_argument('--path_to_image_not_processed', type = str, required=True, 
                    help='Path to the directory to save images not processed')
args = parser.parse_args()

target_aspect_ratio = 4.0 / 3.3 
aspect_ratio_tolerance = 0.15 

desired_width = 420
desired_height = 450

cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)

folder_exist = os.path.exists(args.path_to_saved_ROI)
if not folder_exist:
    os.makedirs(args.path_to_saved_ROI)
    print("A new directory to save the ROI images has been created!")
    
folder_exist_2 = os.path.exists(args.path_to_image_not_processed)
if not folder_exist_2:
    os.makedirs(args.path_to_image_not_processed)
    print("A new directory to save the ROI images has been created!")    


for file in os.listdir(args.path_to_PNG):
    if file.endswith(".png"):
        file_name, _ = os.path.splitext(file)
        img_path = os.path.join(args.path_to_PNG, file)
        img = cv2.imread(img_path)

        if img.shape[0] < img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_height, img_width = img.shape[:2]

        superior_right_quadrant = (img_width // 2, 0, img_width, img_height // 2)
        left_lower_quadrant = (0, img_height // 2, img_width // 2, img_height)

        detected_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            x, y, w, h = cv2.boundingRect(contour)

            in_superior_right = (
                x >= superior_right_quadrant[0] and
                x + w <= superior_right_quadrant[2] and
                y >= superior_right_quadrant[1] and
                y + h <= superior_right_quadrant[3]
            )

            in_left_lower = (
                x >= left_lower_quadrant[0] and
                x + w <= left_lower_quadrant[2] and
                y >= left_lower_quadrant[1] and
                y + h <= left_lower_quadrant[3]
            )

            if in_superior_right or in_left_lower:
                if area > 220000 and perimeter < 3500:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    x, y, w, h = cv2.boundingRect(box)
                    aspect_ratio = float(w) / h if w > h else float(h) / w
                    
                    if abs(aspect_ratio - target_aspect_ratio) < aspect_ratio_tolerance:
                        detected_contours.append((contour, area, box))
        
        if detected_contours:
            smallest_contour = min(detected_contours, key=lambda c: c[1])
            selected_box = smallest_contour[2]
            
            x, y, w, h = cv2.boundingRect(selected_box)
            center, angle = cv2.minAreaRect(smallest_contour[0])[0], cv2.minAreaRect(smallest_contour[0])[2]
            
            if angle > 10:
                angle = angle + 270

            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))

            roi = rotated_image[y + 75:y + h - 65, x + 15:x + w - 15]
            
            resized_roi = cv2.resize(roi, (desired_width, desired_height))
            
            cv2.drawContours(img, [selected_box], 0, (0, 255, 0), 2)

            cv2.imshow('Img', img)
            cv2.imshow('ROI', resized_roi)
            cv2.imshow('Thresh', thresh)
            cv2.imshow('Gray', img_gray)

            save_path = os.path.join(args.path_to_saved_ROI, f"{file_name}.png")
            cv2.imwrite(save_path, resized_roi)
            
            cv2.waitKey(0)
        else:
            print(f"No matching rectangle found for {file}.")
            new_file_path = os.path.join(args.path_to_image_not_processed, file_name + '.png')
            shutil.copy2(img_path, new_file_path)


        
        detected_contours = []

cv2.destroyAllWindows()