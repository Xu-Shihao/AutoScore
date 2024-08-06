import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
import os
import cv2
from as_src.llm_service import call_gpt4_vision, call_claude
from as_src.blank_segmentation import Model as cloze_segment_model
from log.log_schedular import setup_logger
import numpy as np
from scipy.spatial.distance import cdist
from as_src.plot import annotate_image
import paddleocr
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from as_src.llm_service import parse_llm_json
import pickle

logger = setup_logger()

class model:

    def __init__(self, language:str="en"):
        self.CLS_ID_NAME_MAP = {
            0: 'student_id',
            1: 'subjective_problem',
            2: 'fillin_problem',
            3: 'objective_problem'
        }
        self.seg_model = YOLO(model='/home/AutoScore/OCRAutoScore/segmentation/Layout4Card/runs/detect/train5/weights/best.pt')
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang=language)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", device=self.device)
        self.debug = False

    def paper_segmentation(self, img_path:str, output_img_folder:str):
        
        pickle_file = os.path.join(output_img_folder, "segmentation_data.pkl")
    
        if os.path.exists(output_img_folder):
            logger.info("Folder exists, attempting to load pickle file...")
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info("Loaded data from pickle file.")
                return data['bounding_boxes'], data['cls_names']
        
        imgs = []
        img = cv2.imread(img_path)
        imgs += [img]
        results =  self.seg_model.predict(source=imgs, save=False, imgsz=640)
        if output_img_folder:
            # save overall segment result
            os.makedirs(output_img_folder, exist_ok=True)

            # read bounding boxes and select the proper ones based on pixel range
            pixel_ranges = [[209, 1054, 420, 1206], 
                            [636, 1059, 851, 1210], 
                            [209, 1289, 460, 1428], 
                            [209, 1538, 460, 1733], 
                            [1126, 136, 1411, 353], 
                            [1126, 1036, 1411, 1219], 
                            [2230, 151, 2479, 284]]
            
            # get bounding box and class name
            bounding_boxes = []
            cls_names = []
            for result in results:
                for box in result.boxes:
                    cls_id = box.cls.cpu().numpy()[0]
                    x,y,x1,y1 = box.xyxy.cpu().numpy()[0]
                    cls_name = self.CLS_ID_NAME_MAP[cls_id]
                    bounding_boxes.append([x,y,x1,y1])
                    cls_names.append(cls_name)

            # select the bounding box in the pixel range
            bounding_boxes, cls_names = self.filter_bounding_boxes(bounding_boxes, result.boxes.conf.cpu().numpy(), cls_names, pixel_ranges)
            
            for i, name in enumerate(cls_names):
                if name == 'objective_problem':
                    # 根据固定的选择题高度(150 pixel)过滤掉检测出题目的部分
                    if bounding_boxes[i][3] - bounding_boxes[i][1] > 155:
                        bounding_boxes[i][1] = bounding_boxes[i][3] - 155

            annotate_image(img_path, bounding_boxes, cls_names, os.path.join(output_img_folder, "paper_segmentation.jpg"))

            id = 0
            for (x, y, x1, y1), cls_name in zip(bounding_boxes, cls_names):
                cropped_img = img[int(y):int(y1), int(x):int(x1)]
                output_path = os.path.join(output_img_folder, cls_name + "_" + str(id) + "_" + 
                            os.path.splitext(os.path.basename(img_path))[0] + ".jpg")
                cv2.imwrite(output_path, cropped_img)
                id += 1

        # Save bounding_boxes and cls_names to pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump({'bounding_boxes': bounding_boxes, 'cls_names': cls_names}, f)
        logger.info(f"Saved segmentation data to {pickle_file}")
        
        return bounding_boxes, cls_names

    def cloze_problem_segmentation(self, image_path:str, output_img_folder:str):

        # if os.path.exists(output_img_folder):
        #     return None

        model = cloze_segment_model(output_folder_name = output_img_folder, debug=False)
        res = model.process(image_path, os.path.splitext(os.path.basename(image_path))[0])

        return res

    def cloze_problem_solving(self, image_path:str, ground_truth:str):

        prompt = f"""# Roal
You are an professional examiner. 

# Skill
You can recognize the student handwritten answer in the image and judge whether it is correct.

# Goal
Please first recognize the handwritten numbers or equations or text in the image and determine if it matches the expected answer.
You should consider variations in wording or phrasing in mathmatics that convey the same meaning, for example 0.02 and 2% are equal, '0.05' and '$\\frac{{1}}{{20}}$' are same.

# The expected answer
'{ground_truth}'

# Constrain
1. If the answer in the picture is incomplete or outside the picture, the answer is wrong.
2. If you can not recognize the content, content is empty, or you need human to decide, you should choose Unclear!!
3. You CANNOT guess the answer based on the content in #The expected answer. 

## Example
- Expected answer: $0.0255$, Recognized letters: 2.55%, judgment result: Correct, Reason: The recognized letters "2.55%" matches to the expected answer, which is "2.55%".
- Expected answer: $\\frac{{1}}{{20}}$, Recognized letters: 0.05, judgment result: Correct, Reason: The recognized letters "0.05" matches to the expected answer.
- Expected answer: $0.0255$, Recognized letters: 2.5%, judgment result: Incorrect, Reason: The recognized letters "2.5%" doesn't match to the expected answer "2.55%".
- Expected answer: round Truth: $0.0255$, Recognized letters: Unclear, judgment result: Unclear, Reason: The content is empty or the content too blur to make sure what it is.


# output format
Provide your response in JSON format with the following structure:
{{
    "observation": "Describe what you see in the image, focusing on the handwritten text.",
    "thought": "Analyze how the observed text compares to the expected answer. Consider potential variations in wording or phrasing.",
    "image content": "**ONLY** the content of the handwritten text."
    "action": "Decide to choose which one: 'Correct' if the answer matches or conveys the same meaning, 'Incorrect' if it doesn't, or 'Unclear' if the image is unclear or text unreadable. Briefly explain your decision.",
    "judgment": "Choose **ONLY** one: Correct, Incorrect, or Unclear."
}}
"""
        return call_gpt4_vision(image_path, prompt)

    def cloze_problem_solving_3step(self, image_path: str, ground_truth: str):
        # Step 1: Recognize handwritten content
        recognition_prompt = """
        # Role
        You are a professional handwriting recognition expert.

        # Skill
        You can accurately recognize handwritten numbers, equations, or text in images.

        # Goal
        Please recognize and transcribe the handwritten content in the provided image.

        # Output format
        Provide your response in JSON format with the following structure:
        {
            "observation": "Describe what you see in the image, focusing on the handwritten text.",
            "image_content": "ONLY the content of the handwritten text."
        }
        """
        recognition_result = call_gpt4_vision(image_path, recognition_prompt)
        recognition_result = parse_llm_json(recognition_result)
        
        # Step 2: Compare and correct recognized content
        comparison_prompt = f"""
        # Role
        You are a professional examiner and content verifier.

        # Skill
        You can compare recognized content with an expected answer and correct any discrepancies.

        # Goal
        Compare the recognized content with the expected answer. If there are minor discrepancies, correct them based on your knowledge of mathematical equivalences.

        # The expected answer
        '{ground_truth}'

        # Recognized content
        {recognition_result['image_content']}

        # Constrain
        Consider variations in wording or phrasing in mathematics that convey the same meaning, for example, 0.02 and 2% are equal, '0.05' and '$\\frac{{1}}{{20}}$' are the same.

        # Output format
        Provide your response in JSON format with the following structure:
        {{
            "thought": "Analyze how the recognized text compares to the expected answer. Consider potential variations in wording or phrasing.",
            "corrected_content": "ONLY the content of recognized content here, corrected if necessary based on mathematical equivalences."
        }}
        """
        comparison_result = call_gpt4_vision(image_path, comparison_prompt)
        comparison_result = parse_llm_json(comparison_result)

        # Step 3: Judge correctness
        judgment_prompt = f"""
        # Role
        You are a professional examiner.

        # Skill
        You can judge whether a given ocr recognized answer is correct based on an expected answer.

        # Goal
        Determine if the recognized content matches the expected answer and provide a judgment.

        # The expected answer
        '{ground_truth}'

        # Recognized content
        {comparison_result['corrected_content']}

        # Constrain
        1. If the answer is incomplete or would be outside the picture, the answer is wrong.
        2. If the content is empty or you need humans to decide, you should choose Unclear.

        # Output format
        Provide your response in JSON format with the following structure:
        {{
            "thought": "Describe how the recognized text compares to the expected answer. Consider potential variations in wording or phrasing.",
            "action": "Decide to choose which one: 'Correct' if the answer matches or conveys the same meaning, 'Incorrect' if it doesn't, or 'Unclear' if the content is unclear. Briefly explain your decision.",
            "judgment": "Choose ONLY one: Correct, Incorrect, or Unclear."
        }}
        """
        judgment_result = call_gpt4_vision(image_path, judgment_prompt)
        judgment_result = parse_llm_json(judgment_result)

        # Combine results into final output
        final_output = {
            "observation": recognition_result['observation'],
            "thought": judgment_result['thought'],
            "image content": comparison_result['corrected_content'],
            "action": judgment_result['action'],
            "judgment": judgment_result['judgment']
        }
        
        return final_output

    def cloze_problem_checking(self, image_path:str, ocr_text:str, ground_truth:str):

        prompt = f"""# Roal
You are a professional OCR recognition result checker. 

# Goal 
You can check whether the recognized text in the image is consistent with the ground truth.
You should consider variations in wording or phrasing in mathmatics that convey the same meaning, for example 0.02 and 2% are equal, '0.05' and '$\\frac{{1}}{{20}}$' are same.

# The ground truth answer
'{ground_truth}'

# The recognized answer
'{ocr_text}'

# Constrain
1. The handwriting may be difficult to read. You need to refer to the recognized answer and the ground truth answer to determine whether the answer is correct.
2. If the answer in the picture is incomplete or outside the picture, the answer is wrong.
3. If you can not determine whether the recognized answer is correct or not and need human to decide, you should choose "Unclear"!!

## Example
- Ground Truth: $0.0255$, Recognized letters: 2.55%, judgment result: Correct, Reason: The recognized letters "2.55%" matches to the ground truth, which is "2.55%".
- Ground Truth: $\\frac{{1}}{{20}}$, Recognized letters: 0.05, judgment result: Correct, Reason: The recognized letters "0.05" matches to the ground truth.
- Ground Truth: $0.0255$, Recognized letters: 2.5%, judgment result: Incorrect, Reason: The recognized letters "2.5%" doesn't match to the ground truth "2.55%".


# output format
Provide your response in JSON format with the following structure:
{{
    "observation": "Describe what you see in the image and the ocr recognized text.",
    "thought": "Analyze how the observed text compares to the expected answer. Consider potential variations in math wording or phrasing.",
    "image content": "**ONLY** the content of the recognized text: {ocr_text}"
    "action": "Decide to choose which one: 'Correct' if the answer matches or conveys the same meaning, 'Incorrect' if it doesn't, or 'Unclear' if the image is unclear or text unreadable. Briefly explain your decision.",
    "judgment": "Choose **ONLY** one: Correct, Incorrect, or Unclear."
}}
"""
        return call_gpt4_vision(image_path, prompt)
    

    def filter_bounding_boxes(self, bounding_boxes, confidences, labels, pixel_ranges):
        """
        Filter bounding boxes based on input pixel ranges.
        
        Args:
        bounding_boxes (list): List of bounding boxes in format (x1, y1, x2, y2)
        confidences (list): List of confidence scores for each bounding box
        labels (list): List of labels for each bounding box
        pixel_ranges (list): List of pixel ranges in format (x1_min, y1_min, x1_max, y1_max)
        
        Returns:
        tuple: (filtered_boxes, filtered_labels)
        filtered_boxes (list): Filtered bounding boxes
        filtered_labels (list): Corresponding labels of filtered bounding boxes
        """
        filtered_boxes = []
        filtered_labels = []
        
        for pixel_range in pixel_ranges:
            x1_min, y1_min, x1_max, y1_max = pixel_range
            
            # Filter boxes within the current pixel range
            valid_boxes = [
                (i, box) for i, box in enumerate(bounding_boxes)
                if x1_min <= box[0] <= x1_max and y1_min <= box[1] <= y1_max
            ]
            
            if valid_boxes:
                # If multiple boxes are found, choose the one with highest confidence
                best_box_index, best_box = max(valid_boxes, key=lambda x: confidences[x[0]])
                filtered_boxes.append(best_box)
                filtered_labels.append(labels[best_box_index])
            else:
                # If no box is found, find the nearest box
                upper_left_corners = np.array([[box[0], box[1]] for box in bounding_boxes])
                target_point = np.array([(x1_min + x1_max) / 2, (y1_min + y1_max) / 2])
                
                distances = cdist([target_point], upper_left_corners)[0]
                nearest_box_index = np.argmin(distances)
                nearest_box = bounding_boxes[nearest_box_index]
                nearest_label = labels[nearest_box_index]
                
                # filtered_boxes.append(nearest_box)
                # filtered_labels.append(nearest_label)
                logger.error(f"No bounding box found in range {pixel_range}. Nearest box: {nearest_box}, Label: {nearest_label}")
        
        return filtered_boxes, filtered_labels
    
    def recognize_text(self, _img:Image):
        """
        Predict the text from image
        :parameter img: image, type: np.ndarray
        :return: result, type: tuple{location: list, text: str}
        """
        img = np.array(_img)
        result = self.ocr.ocr(img)
        if self.debug:
            print(result)
        if not result[0] or len(result[0]) == 0:
            return None
        else:
            location = result[0][0][0]
            text = result[0][0][1][0]
            return (location, text)

    def judge_with_clip(self, _answer:str, _predict:str, _img:Image):
        """
        Use clip to judge which one is more similar to the Image
        :parameter answer: the answer text, type: str
        :parameter predict: the predict text, type: str
        :parameter img: image, type: np.ndarray
        :return: result, the index of the more similar text, type: int
        """
        image = _img
        inputs = self.clip_processor(text=[f"A picture with the text \"{_answer}\"", f"A picture with the text \"{_predict}\"",
                                 "A picture with the other text"], images=image, return_tensors="pt", padding=True)
        inputs.to(self.device)

        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        if self.debug:
            print(probs)
        index = torch.argmax(probs, dim=1)
        return index