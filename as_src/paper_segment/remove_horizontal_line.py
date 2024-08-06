import cv2
import numpy as np

def detect_and_remove_horizontal_lines(input_path, output_path, x_start, x_end, min_line_length=50):
    # Read the image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=min_line_length, maxLineGap=10)
    
    if lines is None:
        cv2.imwrite(output_path, img)
        return

    # Filter and merge horizontal lines
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) <= 5:  # Consider as horizontal if y difference is 5 pixels or less
            horizontal_lines.append((min(x1, x2), max(x1, x2), y1))
    
    # Sort lines by y-coordinate
    horizontal_lines.sort(key=lambda x: x[2])
    
    # Merge lines that are close in y-coordinate
    merged_lines = []
    for line in horizontal_lines:
        if not merged_lines or abs(line[2] - merged_lines[-1][2]) > 5:
            merged_lines.append(line)
        else:
            last_line = merged_lines[-1]
            merged_lines[-1] = (min(last_line[0], line[0]), max(last_line[1], line[1]), (last_line[2] + line[2]) // 2)

    # Filter lines based on x_start and x_end
    lines_to_remove = [line for line in merged_lines if line[0] <= x_start and line[1] >= img.shape[1] + x_end]

    # Remove the lines with smart filling
    result = img.copy()
    for x1, x2, y in lines_to_remove:
        # Expand the range of removal
        y_start = max(0, y - 10)
        y_end = min(img.shape[0], y + 5)
        
        for x in range(x1, x2):
            # Check pixels above and below
            above = gray[max(0, y_start-1), x]
            below = gray[min(img.shape[0]-1, y_end+1), x]
            
            # If both above and below are black (or very dark), keep the original pixel
            if above < 50 and below < 50:
                continue
            
            # Otherwise, fill with white
            result[y_start:y_end, x] = [255, 255, 255]  # White color in BGR

    # Save the result
    cv2.imwrite(output_path, result)
    return 

if __name__ == "__main__":
    # Usage
    input_path = '/home/segmentation/g2dh52E_01000305_01A.png'
    output_path = 'image_without_horizontal_line.png'
    x_start = 100  # Adjust these values based on your specific requirements
    x_end = -1000
    min_line_length = 50  # Minimum length of lines to consider
    detect_and_remove_horizontal_lines(input_path, output_path, x_start, x_end, min_line_length)