import docx2txt
import os
import docx
import os
from docx_equation.docx import convert_to_html
from docxlatex import Document
import re

def load_docx(file_path:str, image_saved_folder=None):

    # extract text and write images in /tmp/img_dir
    text = docx2txt.process(file_path, image_saved_folder) 

    return text

def docx2doc(file_path, output_file_path):
    doc = docx.Document(file_path)
    doc.save(output_file_path)

def load_doclatex(file_path):

    # word文件的公式需要在word里转为latex格式
    docx = Document(file_path)
    text = docx.get_text()
    return text


def extract_questions_and_answers(text):
    # 分割不同题型
    sections = re.split(r'([一二三四五六七八九十]+[､、].*?[:：]本题.*?)\n', text)[1:]
    
    result = []
    for i in range(0, len(sections), 2):
        section_title = sections[i].strip().split("､")[1].split("：")[0]
        section_content = sections[i+1]
        
        # 提取该题型下的所有题目
        questions = re.findall(r'(\d+)\. .*?【答案】(.*?)【解析】(.*?)(?=\d+\. |$)', section_content, re.DOTALL)
        
        section_dict = {}
        for num, answer, analysis in questions:
            # 处理选择题和填空题
            if '（' not in answer and '见解析' not in answer.lower():
                answers = answer.split('##')
                answers = [a.strip().replace('\n', '').replace(' ', '') for a in answers]
                section_dict[num] = answers[0] if len(answers) == 1 else answers
            else:
                # 处理解答题
                if '见解析' in answer.lower():
                    section_dict[num] = [analysis.strip()]
                else:
                    section_dict[num] = [answer]
        
        result.append([section_title, section_dict])

    return result


if __name__ == "__main__":
    """
    用于测试函数
    """
    file_path = r"D:\code\AutoScore\AutoScore\data\g2dh2\期末数学试题_答案_convert.docx"
    image_saved_folder=r'D:\code\AutoScore\AutoScore\data\g2dh2\ans_doc_imgs'
    os.makedirs(image_saved_folder, exist_ok=True)
    # text = load_docx(file_path = file_path, image_saved_folder=image_saved_folder)
    # print(text[:400])

    text = load_doclatex(file_path)
    print(extract_questions_and_answers(text)[:3])
    