import cv2
import numpy as np
import os
import argparse

points = []
resized_img = None

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

def apply_process(img, points, desired_width, desired_height):
    if len(points) != 4:
        print("Selecciona 4 puntos.")
        return None
    
    selected_box = np.array(points, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(selected_box)

    rect = cv2.minAreaRect(selected_box)
    center, angle = rect[0], rect[2]

    if angle > 10:
        angle += 270

    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))

    roi = rotated_image[y + 75:y + h - 65, x + 15:x + w - 15]

    resized_roi = cv2.resize(roi, (desired_width, desired_height))

    return resized_roi

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            points = []  

            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            original_img = img.copy()
            
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 1285,1832)

            cv2.imshow("Image", img)
            cv2.setMouseCallback("Image", select_points)

            print(f"Selecciona 4 puntos en la imagen: {filename}")
            cv2.waitKey(0)

            if len(points) == 4:
                result = apply_process(original_img, points, 420, 450)
                if result is not None:
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, result)
                    print(f"Imagen procesada guardada en: {output_path}")
            else:
                print(f"No se seleccionaron los 4 puntos en la imagen: {filename}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa imágenes seleccionando 4 puntos con el mouse.")
    parser.add_argument("input_folder", type=str, help="Carpeta donde están las imágenes.")
    parser.add_argument("output_folder", type=str, help="Carpeta donde se guardarán las imágenes procesadas.")

    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder)

