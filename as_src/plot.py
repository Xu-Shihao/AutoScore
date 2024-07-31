from PIL import Image, ImageDraw, ImageFont
import numpy as np
from log.log_schedular import setup_logger

logger = setup_logger()

def annotate_image(image_path, bounding_boxes, labels, output_path):
    """
    Annotate an image with bounding boxes and labels.
    
    :param image_path: Path to the input image
    :param bounding_boxes: List of bounding boxes in format [x1, y1, x2, y2]
    :param labels: List of labels corresponding to each bounding box
    :param output_path: Path to save the annotated image
    """
    # Open the image
    image = Image.open(image_path)

    # Convert to RGB if the image is in grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')

    draw = ImageDraw.Draw(image)
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    # Define colors for different labels
    color_map = {}
    
    for box, label in zip(bounding_boxes, labels):
        # Assign a color to each unique label
        if label not in color_map:
            color_map[label] = tuple(np.random.randint(0, 255, 3))
        
        color = color_map[label]
        
        # Draw the bounding box
        draw.rectangle(box, outline=color, width=5)
        
        # Draw the label
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)
    
    # Save the annotated image
    image.save(output_path)
    logger.info(f"Annotated image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    bounding_boxes = [
        [100, 100, 200, 200],
        [300, 300, 400, 400],
        # Add more bounding boxes as needed
    ]
    labels = [
        "subjective_question 0.96",
        "objective_question 0.82",
        # Add more labels corresponding to each bounding box
    ]
    output_path = "path/to/save/annotated_image.jpg"
    
    annotate_image(image_path, bounding_boxes, labels, output_path)