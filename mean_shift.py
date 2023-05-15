import os
import numpy as np
import random
import cv2
import argparse
import time

def eucledian_distance(a, b):
    return np.linalg.norm(a-b)

def find_peak_opt(data, i, r, c, threshold):
    point = data[:, i]
    cpts = np.zeros(data.shape[1], dtype=bool)

    while True:
        distance = np.linalg.norm(data - point.reshape(-1, 1), axis=0)
        window_points = np.where(distance <= r)[0]

        if len(window_points) == 0:
            return point, cpts

        new_mean = np.mean(data[:, window_points], axis=1)

        if np.linalg.norm(new_mean - point) < threshold:
            distances = np.linalg.norm(data[:, window_points] - new_mean[:, np.newaxis], axis=0)
            cpts[window_points[distances <= r/c]] = True
            return new_mean, cpts

        point = new_mean

def mean_shift_opt(data, r, c, threshold):
    labels = np.zeros(data.shape[1])
    peaks = []

    for i in range(data.shape[1]):
        if i * 10 % data.shape[1] == 0:
            print("Progress: ", i / data.shape[1] * 100, "%")
        if labels[i] == 0:
            point = data[:, i]
            peak_found = False
            for j in range(len(peaks)):
                dist = eucledian_distance(point, peaks[j])
                if dist <= r:
                    labels[i] = j + 1
                    peak_found = True
                    break

            if not peak_found:
                peak, cpts = find_peak_opt(data, i, r, c, threshold)
            else:
                continue

            # Check if the peak is similar to any existing peaks
            for j in range(len(peaks)):
                if eucledian_distance(peak, peaks[j]) <= r/2:
                    labels[i] = j + 1
                    peak_found = True
                    break

            # If the peak is not similar to any existing peaks, assign a new label
            if not peak_found:
                labels[i] = len(peaks) + 1
                peaks.append(peak)
                labels[cpts] = len(peaks)

    return labels, np.array(peaks).T

def assign_segment_colors(labels, image):
    num_segments = int(np.max(labels)) + 1
    colors = np.zeros((num_segments, 3), dtype=np.uint8)

    for segment in range(num_segments):
        mask = labels == segment
        segment_pixels = image[mask]
        colors[segment] = np.mean(segment_pixels, axis=0)

    colored_labels = colors[labels.astype(int)]
    return colored_labels

def showImage(window_text, image):
    cv2.imshow(window_text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveImage(image, image_path, features=3):
    image_name = image_path.split("/")[-1]
    image_name = image_name.split(".")[0]
    root_filename = "output/segmented_" + str(features) + "_" + image_name
    filename = root_filename + ".jpg"
    cnt = 1
    while True:
        if os.path.exists(filename):
            filename = filename.split(".jpg")[0]
            filename = filename[:len(filename) - 2]
            filename = filename + "_" + str(cnt) + ".jpg"
            cnt += 1
        else:
            break
    print("Saving image to: ", filename)
    cv2.imwrite(filename, image)
    return 1

def imageSegmentation(image_path, r, c, feature, scale, threshold, save):
    image = cv2.imread(image_path)
    showImage("Original Image", image)

    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    showImage("Resized Image", resized)

    lab_image = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    showImage("LAB Image", lab_image)

    start = time.time()
    # Reshape the image into a 3-by-p array
    colours = lab_image.reshape((-1, 3)).T.astype(np.float32)

    if feature == 3:
        data = colours
    elif feature == 5:
        # Extract spatial coordinates
        height, width, _ = lab_image.shape
        x_coords, y_coords = np.indices((height, width)).astype(np.float32)

        # Reshape the coordinates to match the shape of the color values
        x_coords = x_coords.reshape(-1)
        y_coords = y_coords.reshape(-1)

        # Create the 5-by-p data matrix
        data = np.vstack((colours, x_coords, y_coords))

    print("data shape: ", data.shape)
    # Apply the Mean-shift algorithm
    labels, peaks = mean_shift_opt(data, r, c, threshold)

    segmented_image = labels.reshape(lab_image.shape[:2])
    # Assign colors to segments
    colored_image = assign_segment_colors(segmented_image, lab_image)
    segmented_image_rgb = cv2.cvtColor(colored_image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    rescaled_image = cv2.resize(segmented_image_rgb, (image.shape[1], image.shape[0]))

    end = time.time()
    print("Time taken: ", end - start, "seconds")

    # Visualize the segmented image
    if save:
        _ = saveImage(rescaled_image, image_path, feature)
    else:
        showImage("Segmented Image", rescaled_image)

def main():
    parser = argparse.ArgumentParser(description='Mean Shift Image Segmentation')

    # Add arguments
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--radius', type=int, default=30, help='Window radius for mean shift')
    parser.add_argument('-c', type=int, default=4, help='Constant value for speedup')
    parser.add_argument('--features', type=int, default=3, help='3 or 5 dimension feature vector')
    parser.add_argument('--scale', type=float, default=0.3, help='scale your image')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for merging peaks')
    parser.add_argument('--save', type=bool, default=False, help='Save image')

    args = parser.parse_args()
    imageSegmentation(args.image, args.radius, args.c, args.features, args.scale, args.threshold, args.save)


if __name__ == '__main__':
    main()