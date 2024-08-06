import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

def filter_bounding_boxes(bounding_boxes, confidences, pixel_ranges):
    """
    Filter bounding boxes based on input pixel ranges.
    
    Args:
    bounding_boxes (list): List of bounding boxes in format (x1, y1, x2, y2)
    confidences (list): List of confidence scores for each bounding box
    pixel_ranges (list): List of pixel ranges in format (x1_min, y1_min, x1_max, y1_max)
    
    Returns:
    tuple: (filtered_boxes, filtered_labels)
    filtered_boxes (list): Filtered bounding boxes
    """
    filtered_boxes = []

    for pixel_range in pixel_ranges:
        x1_min, y1_min, x1_max, y1_max = pixel_range[0]
        x2_min, y2_min, x2_max, y2_max = pixel_range[1]
        
        # Filter boxes within the current pixel range
        valid_boxes = [
            (i, box) for i, box in enumerate(bounding_boxes)
            if x1_min <= box[0] <= x1_max and y1_min <= box[1] <= y1_max and x2_min <= box[0] + box[2] <= x2_max and y2_min <= box[1] + box[3] <= y2_max
        ]
        
        if valid_boxes:
            # If multiple boxes are found, choose the one with highest confidence
            best_box_index, best_box = max(valid_boxes, key=lambda x: confidences[x[0]])
            filtered_boxes.append(best_box)
        elif bounding_boxes:
            # If no box is found, find the nearest box
            upper_left_corners = np.array([[box[0], box[1]] for box in bounding_boxes])
            target_point = np.array([(x1_min + x1_max) / 2, (y1_min + y1_max) / 2])
            
            distances = cdist([target_point], upper_left_corners)[0]
            nearest_box_index = np.argmin(distances)
            nearest_box = bounding_boxes[nearest_box_index]
            
            # filtered_boxes.append(nearest_box)
            # filtered_labels.append(nearest_label)
            print(f"No bounding box found in range {pixel_range}. Nearest box: {nearest_box}")
        else:
            print(f"No bounding box found in range {pixel_range}.")

    return filtered_boxes

def segment_exam_paper(image_path, output_path = None, side = None):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
    
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine horizontal and vertical lines
    grid = cv2.add(horizontal_lines, vertical_lines)
    
    # Dilate the grid to connect nearby lines
    kernel = np.ones((7,7), np.uint8)
    grid = cv2.dilate(grid, kernel, iterations=2)
    
    # 假设你已经有了一个二值化图像 grid
    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # # hierarchy[i][3] 表示父轮廓的索引。-1 表示没有父轮廓，即它是最外层轮廓
    inner_contours = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] != -1]

    # Filter contours
    height, width = img.shape[:2]
    min_area = 0.005 * height * width
    max_area = 0.5 * height * width
    
    valid_contours = []
    for contour in inner_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if min_area < area < max_area and 0.2 < aspect_ratio < 5:
            valid_contours.append((x, y, w, h, area))
    
    def sort_contours(cnts):
        # Determine the number of columns (adjust this based on your specific image)
        num_columns = 3
        column_width = width // num_columns
        
        # Group contours into columns
        columns = [[] for _ in range(num_columns)]
        for cnt in cnts:
            x, y, w, h, area = cnt
            column_index = x // column_width
            columns[column_index].append(cnt)
        
        # Sort each column
        for column in columns:
            column.sort(key=lambda c: (c[1], c[0], c[4]))  # Sort by y, then x, then area
        
        # Flatten the sorted columns
        return [item for column in columns for item in column]

    sorted_contours = sort_contours(valid_contours)

    # select based on pre-defined region
    # For *A.TIF
    if side != None:
        if side == 0:  # 第一面试卷
            pixel_ranges = [[[191, 1065, 267, 1134], [1101, 1281, 1165, 1341]],
                            [[186, 1267, 271, 1357], [1101, 1530, 1165, 1581]], 
                            [[186, 1525, 262, 1611], [1101, 2048, 1165, 2114]], 
                            [[1126, 136, 1411, 353], [2205, 1045, 2277, 1104]], 
                            [[1126, 1036, 1411, 1219], [2205, 2047, 2277, 2118]], 
                            [[2230, 151, 2479, 284], [3131, 2062, 3205, 2123]]]
            sorted_contours = filter_bounding_boxes(sorted_contours, [1 for _ in sorted_contours], pixel_ranges)

        elif side == 1: # 第二面试卷
            pixel_ranges = [[[154, 129, 263, 233], [1111, 2005, 1204, 2096]], 
                            [[1152, 129, 1261, 233], [2129, 2005, 2222, 2096]], 
                            [[2175, 129, 2289, 233], [3090, 2005, 3185, 2096]]]
            sorted_contours = filter_bounding_boxes(sorted_contours, [1 for _ in sorted_contours], pixel_ranges)

        else:
            raise f"Side '{side}' is not support at this time."

    # if output_path:
    #     # Draw rectangles and number them
    #     for i, (x, y, w, h, _) in enumerate(sorted_contours):
    #         color = (0, 255, 0)  # Green color
    #         thickness = 2
    #         cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    #         # Add question number
    #         cv2.putText(img, f'Q{i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    #     # Save or display the result
    #     cv2.imwrite(output_path, img)

    # Create list of results
    results = []
    for i, (x, y, w, h, _) in enumerate(sorted_contours):
        results.append({
            "area_number": i + 1,
            "coordinates": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
        })
    
    return results

def cut_and_save_image(input_path, output_path, coordinates):
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Extract coordinates
            x, y, w, h = coordinates
            
            # Crop the image
            cropped_img = img.crop((x, y, x+w, y+h))
            
            # Save the cropped image
            cropped_img.save(output_path)
        
        print(f"Image successfully cropped and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":

    # Usage
    image_path = 'example_image\g2dh229_01081313_03B.TIF'
    output_path = 'example_image\g2dh229_01081313_03B_segement.jpg'
    segmentation_results = segment_exam_paper(image_path, output_path=output_path, side=1)

    # Print results
    for result in segmentation_results:
        print(f"Area {result['area_number']}: {result['coordinates']}")
