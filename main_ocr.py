import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os, re
import glob
import pandas as pd
from as_src.llm_service import parse_llm_json
from as_src.docx_loader import load_doclatex, extract_questions_and_answers
from log.log_schedular import setup_logger
import numpy as np
from PIL import Image
from as_src.model import model

logger = setup_logger()

def query_score_by_image(image_name, merged_df, log = False):
    # 查找对应的行
    result = merged_df[merged_df['图像名称'] == image_name]
    
    if result.empty:
        return "未找到该图像名称对应的成绩信息"
    
    # 获取成绩信息
    exam_number = int(result['考号'].values[0])
    total_score = result['卷面得分'].values[0]
    
    # 获取选择题成绩
    choice_scores = result.filter(regex='^XZ-').values[0]
    
    # 获取其他题目成绩
    other_scores = result.filter(regex='^(?!XZ-)\\d+$').values[0]
    
    # 构建返回信息
    # info = f"考号: {exam_number}\n总分: {total_score}\n"
    # info += "选择题成绩: " + ", ".join([f"{i+1}:{score}" for i, score in enumerate(choice_scores)]) + "\n"
    # info += "填空题目成绩: " + ", ".join([f"{i+13}:{score}" for i, score in enumerate(other_scores[:4])]) + "\n"
    # info += "解答题目成绩: " + ", ".join([f"{i+17}:{score}" for i, score in enumerate(other_scores[4:])])
    # logger.info(info)
    
    return list(choice_scores) + list(other_scores)

def extract_number(filename):
    # Extract the number from the filename
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 1  # Return 1 if no number is found

def accuracy_cal(file_path):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Convert 'judgment' column to binary (1 for 'Correct', 0 for others)
    df['judgment_binary'] = (df['judgment'] == 'Correct').astype(int)

    # Convert 'human_score' column to binary (1 for 5, 0 for others)
    df['human_score_binary'] = (df['human_score'] == 5).astype(int)

    # Calculate metrics
    precision = precision_score(df['human_score_binary'], df['judgment_binary'])
    recall = recall_score(df['human_score_binary'], df['judgment_binary'])
    f1 = f1_score(df['human_score_binary'], df['judgment_binary'])

    # Print results
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    return precision, recall, f1

if __name__ == "__main__":
    """
    用于测试函数
    """

    # load answer
    df_1 = pd.read_excel("D:\code\AutoScore\AutoScore\data\g2dh2\文件考号对应关系.xlsx", index_col=None, header=0)
    df_2 = pd.read_excel("D:\code\AutoScore\AutoScore\data\g2dh2\期末数学成绩.xls", index_col=None, header=0)
    merged_df = pd.merge(df_1, df_2, on='考号', how='left')
    
    # initial
    count = 0
    total = 0
    results = []
    auto_score = model()
    gt_file_path = r"D:\code\AutoScore\AutoScore\data\g2dh2\期末数学试题_答案_convert.docx"
    output_path = r"D:\code\AutoScore\AutoScore\output\results_ocr_gpt4.xlsx"

    # load the ground truth
    text = load_doclatex(gt_file_path)
    answers = extract_questions_and_answers(text)
    
    for img_path in sorted(glob.glob(r"D:\code\AutoScore\AutoScore\data\g2dh2\OMR0001\*A.TIF"))[:100]:

        basename = os.path.splitext(os.path.basename(img_path))[0]
        preprocess_folder = os.path.splitext(img_path.replace(r"D:\code\AutoScore\AutoScore\data", r"D:\code\AutoScore\AutoScore\output\preprocess"))[0]
        # img_path=r"D:\code\AutoScore\AutoScore\data\g2dh2\OMR0001\g2dh229_01081312_02A.TIF"

        segmentation_img_folder=os.path.join(preprocess_folder, r"segmentation")
        cloze_segmentation_path=os.path.join(preprocess_folder, r"cloze_segmentation")
        
        ## segment the cloze area
        # auto_score = model()
        auto_score.paper_segmentation(img_path=img_path,  output_img_folder=segmentation_img_folder)
        cloze_image_file_name = sorted(glob.glob(os.path.join(segmentation_img_folder,"subjective_problem_*.jpg")), key=extract_number)[0]
        auto_score.cloze_problem_segmentation(image_path=cloze_image_file_name, output_img_folder=cloze_segmentation_path)

        # judge 
        for title, answer_dic in answers:
            if title == "填空题":
                answer_dic_cloze = answer_dic
        
        cloze_gts = list(answer_dic_cloze.values())
        cloze_imgs = glob.glob(os.path.join(cloze_segmentation_path,f"{os.path.splitext(os.path.basename(cloze_image_file_name))[0]}-*.jpg"))

        # load human judged score
        score_list = query_score_by_image(basename, merged_df)
        cloze_start_number = 4 * len(glob.glob(os.path.join(segmentation_img_folder,f"objective_problem_*{basename}*.jpg")))

        for i, (cloze_img_path, gt) in enumerate(zip(cloze_imgs, cloze_gts)):
            
            # run OCR
            ans = auto_score.recognize_text(Image.open(cloze_img_path))
            
            # if has OCR result
            if not ans:
                respose_json = auto_score.cloze_problem_solving(image_path=cloze_img_path,  ground_truth=gt)
                if not isinstance(respose_json, dict):
                    try:
                        respose_json = parse_llm_json(respose_json)
                    except:
                        logger.error(f"Unable to parse json output. {respose_json}")
                ans = respose_json['image content']
                judg = respose_json["judgment"]
                respose_json = {}
                respose_json['image content'] = ans 
                respose_json["judgment"] = judg
            else:
                ans = ans[1]
                respose_json = auto_score.cloze_problem_checking(image_path=cloze_img_path, ocr_text=ans, ground_truth=gt)
                if not isinstance(respose_json, dict):
                    try:
                        respose_json = parse_llm_json(respose_json)
                    except:
                        logger.error(f"Unable to parse json output. {respose_json}")
                ans = respose_json['image content']
                judg = respose_json["judgment"]
                respose_json = {}
                respose_json['image content'] = ans 
                respose_json["judgment"] = judg

            human_score = score_list[cloze_start_number+i]
            logger.info(f"Ground Truth: {gt}, Recognized letters: {ans}, judgment result: {judg}, human_score: {human_score}")

            if judg == "Correct" and int(human_score) == 5 or judg == "Incorrect" and int(human_score) == 0:
                count += 1
            total += 1

            # Append results to the list
            respose_json['ground_truth'] = gt
            respose_json['basename'] = basename
            respose_json['human_score'] = human_score
            results.append(respose_json)

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Accuracy: {count/total:.2f}")

    accuracy_cal(output_path)
