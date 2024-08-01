import sys
import os
import random
from typing import List, Tuple
import glob
from tqdm import tqdm
from PIL import Image
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from as_src.model import model
from seg_data_generator import segment_exam_paper


def get_image_data() -> List[Tuple[str, List[Tuple[int, int, int, int, str]]]]:
    data_folder = "/home/TX"
    output_folder = "/home/segmentation"
    os.makedirs(output_folder, exist_ok=True)
    
    auto_score = model()
    
    results = []
    for img_path in tqdm(sorted(glob.glob(os.path.join(data_folder, "*/*/*.TIF")))[60000:]):
        
        basename = os.path.splitext(os.path.basename(img_path))[0]
        if img_path.endswith("B.TIF"):
            side = 1
        else:
            side = 0

        output_path = os.path.join(output_folder, os.path.basename(img_path)[:-4] + ".png")
        segmentation_results = segment_exam_paper(img_path, output_path=output_path, side=side)
        try:
            bounding_boxes, cls_names = auto_score.paper_segmentation(img_path=img_path,  output_img_folder=os.path.join(output_folder, "preprocess", basename))
        except:
            bounding_boxes = []
            cls_names = []

        result = []
        if side == 0 and len(segmentation_results) == 6 and cls_names.count('objective_problem') == 3:
            
            """
            |  id   | content  |
            |  ----  | ----  |
            | 0  | 学号 |
            | 1  | 主观题 |
            | 2  | 填空题 |
            | 3  | 客观题 |
            """
            
            for i, name in enumerate(cls_names):
                if name == 'objective_problem':
                    result.append(bounding_boxes[i] + ["1"])
            
            for idx, area in enumerate(segmentation_results):
                if idx == 2:
                    result.append([area["coordinates"]["x"],area["coordinates"]["y"],area["coordinates"]["w"],area["coordinates"]["h"]] + ["2"])
                elif idx > 2:
                    result.append([area["coordinates"]["x"],area["coordinates"]["y"],area["coordinates"]["w"],area["coordinates"]["h"]] + ["3"])
        
        elif side == 1 and len(segmentation_results) == 3:
            
            for idx, area in enumerate(segmentation_results):
                result.append([area["coordinates"]["x"],area["coordinates"]["y"],area["coordinates"]["w"],area["coordinates"]["h"]] + ["3"])

        if side == 0 and len(result) == 7 or side == 1 and len(result) == 3: #第一页7题，第二页3题
            results.append((output_path, result))

    return results


def convert_to_yolo_format(image_width: int, image_height: int, box: Tuple[int, int, int, int, str]) -> str:
    """
    将边界框坐标转换为YOLO格式
    """
    x1, y1, x2, y2, label = box
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    class_id = label
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def create_directories(base_path: str):
    """
    创建必要的目录结构
    """
    for dir_name in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)

def process_data(base_path: str, class_mapping: dict, train_ratio: float = 0.8):
    """
    处理数据并生成YOLO格式的标签文件
    """
    create_directories(base_path)

    image_data = get_image_data()
    
    for image_path, boxes in tqdm(image_data):
        # 决定是训练集还是验证集
        is_train = random.random() < train_ratio
        subset = 'train' if is_train else 'val'

        # 复制图片到目标文件夹
        image_name = os.path.basename(image_path)
        target_image_path = os.path.join(base_path, f'images/{subset}', image_name)
        shutil.copy(image_path, target_image_path)

        # 获取图片尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # 创建并保存标签文件
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(base_path, f'labels/{subset}', label_name)

        with open(label_path, 'w') as f:
            for box in boxes:
                yolo_line = convert_to_yolo_format(image_width, image_height, box)
                f.write(yolo_line + '\n')

def create_yaml_file(base_path: str, class_mapping: dict):
    """
    创建数据集的YAML配置文件
    """
    yaml_content = f"""
path: {base_path}
train: images/train
val: images/val

nc: {len(class_mapping)}
names: {list(class_mapping.values())}
"""
    with open(os.path.join(base_path, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

def main():
    base_path = '/home/dataset'  # 替换为您的数据集路径
    class_mapping = {
        '学号': "0",
        '填空题': "1",
        '客观题': "2",
        '主观题': "3",
    }
    
    process_data(base_path, class_mapping)
    create_yaml_file(base_path, class_mapping)
    print("数据处理完成，YOLOv8格式的数据集已准备就绪。")

if __name__ == "__main__":
    main()