import functools
import sys
import cv2
import os
import numpy as np
import math


class Model:
    def __init__(self, output_folder_name = "./debug/", debug=False):
        self.rects = []  # 记录一张图可以标记分割的矩阵 格式为[x, y, w, h]
        self.crop_img = [] # 保存分割的图片
        self.img = None  # 图片
        self.binary = None # 二值图像
        self.debug = debug  # debug模式
        self.name = ''  # 图片名
        self.output_folder_name = output_folder_name #保存地址

    def process(self, img_path, name):  # 运行过程
        binary = self.__preProcessing(img_path, name)
        horizon = self.__detectLines(binary)
        self.__contourExtraction(horizon)
        result = self.__segmentation()
        self.rects.clear()
        self.img = None
        self.name = ''
        return result

    def __preProcessing(self, img_path, name):  # 图片预处理，输出二值图
        img = cv2.imread(img_path)
        self.img = img
        self.name = name
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (3, 3), 1.5)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh, binary = cv2.threshold(blur, int(_ * 0.95), 255, cv2.THRESH_BINARY)
        self.binary = binary
        return binary

    def process_img(self, img):
        binary = self.__preProcessing_img(img)
        horizon = self.__detectLines(binary)
        self.__contourExtraction(horizon)
        result = self.__segmentation()
        self.rects.clear()
        self.img = None
        return result

    def __preProcessing_img(self, img):  # 图片预处理，输出二值图
        self.img = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh, binary = cv2.threshold(blur, int(_ * 0.95), 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def __detectLines(img):  # 检测水平线
        horizon_k = int(math.sqrt(img.shape[1]) * 1.2)  # w
        # hors_k = int(binary.shape[1]/ 16)  # w
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizon_k, 1))  # 设置内核形状
        horizon = ~cv2.dilate(img, kernel, iterations=1)  # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizon_k / 0.9), 1))
        horizon = cv2.dilate(horizon, kernel, iterations=1)
        return horizon

    def __merge_line(self, line):
        # 合并同一行的矩形
        x = min(rect[0] for rect in line)
        y = min(rect[1] for rect in line)
        right = max(rect[0] + rect[2] for rect in line)
        bottom = max(rect[1] + rect[3] for rect in line)
        return [x, y, right - x, bottom - y]

    def __contourExtraction(self, img, debug=False):  # 轮廓检测
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_y, border_x = img.shape
        # 去除邻近上边界和下边界检测的轮廓
        for cnt in cnts[0]:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < 4 or y > border_y - 8:
                continue
            if self.debug and debug:
                cv2.rectangle(self.img, (x, y), (w + x, h + y), (0, 0, 255), 2)
            self.rects.append([x, y, w, h])
        # 排序
        self.rects = sorted(self.rects, key=functools.cmp_to_key(self.__cmp_rect_r))

        # 合并同一水平线上的直线
        merged_rects = []
        current_line = None
        for rect in self.rects:
            x, y, w, h = rect
            # if w < 150:
            #     continue
            if current_line is None:
                current_line = rect
            elif abs(current_line[1] - y) <= 5:  # 在同一水平线上
                # 合并直线
                current_line[0] = min(current_line[0], x)
                current_line[2] = max(current_line[2] + current_line[0], x + w) - current_line[0]
                current_line[1] = min(current_line[1], y)
                current_line[3] = max(current_line[3], h)
            else:
                merged_rects.append(current_line)
                current_line = rect
        if current_line:
            merged_rects.append(current_line)
        self.rects = merged_rects

        # 标记不相关的轮廓
        pre = None
        idx_lst = []
        for idx, cnt in enumerate(self.rects):
            x, y, w, h = cnt
            if w < 150:
                continue
            if pre is None:
                pre = [x, y, w]
            elif 6 < abs(pre[1] - (y + h / 2)) < 70:  # and 10 < abs(pre[0] - x) < pre[2]
                continue
            pre[1] = y + h / 2
            pre[0] = x
            pre[2] = w
            idx_lst.append(idx)

        # 再次筛选
        self.rects = [self.rects[x] for x in idx_lst]
        self.rects = sorted(self.rects, key=functools.cmp_to_key(self.__cmp_rect))

        contours = self.rects.copy()
        
        # 将检测的水平线扩充成矩形框
        pre_y, pre_h = -1, -1
        for idx, cnt in enumerate(self.rects):
            x, y, w, h = cnt
            if pre_h == -1:
                pre_y = y
                if y > 90:
                    h = y - 70 #去掉第一个填空题上方的题目部分
                    y = 70
                else:
                    h = y - 5  #没有包含题目部分
                    y = 5
                pre_h = h
            else:
                if abs(pre_y - y) < 10:
                    h = pre_h
                    y = max(y - h, 0)
                else:
                    pre_h = abs(y - pre_y) - 10
                    pre_y = y
                    h = pre_h
                    y = pre_y - h
            self.rects[idx] = [x, y, w + 150, h + 15]

        ## 将横线去掉
        # Create a mask for areas to keep
        mask = np.ones(self.binary.shape[:2], dtype=np.uint8) * 255
        
        for contour in contours:
            x, y, w, h = contour
            margin = 1
            # Check for content above and below the line, column by column
            for col in range(x, min(mask.shape[1], x+w)):
                
                above = self.binary[max(0, y-margin):y, col:col+1]
                below = self.binary[y+h:min(self.binary.shape[0], y+h+margin), col:col+1]
                
                # If there's no significant content above or below, mark this column for removal
                if np.min(above) != 0 or np.min(below) != 0:
                    mask[y:y+h, col] = 0
        
        # Apply the mask to the original image
        self.binary = cv2.bitwise_and(self.binary, self.binary, mask=mask)
        
        # Fill removed areas with white
        self.binary[mask == 0] = 255
        self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, (3, 3))

        return contours

    def __segmentation(self):  # 分割
        if self.debug:  # debug模式只标记不分割
            if not os.path.exists('debug'):
                os.mkdir('debug')
            for idx, rect in enumerate(self.rects):
                x, y, w, h = rect
                # h = h + y
                # y = 0
                cv2.rectangle(self.img, (x, y), (w + x, h + y), (255, 0, 255), 2)
                cv2.putText(self.img, str(idx + 1), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imwrite(os.path.join(self.output_folder_name, f'{self.name}.jpg'), self.img)
        else:
            if not os.path.exists(self.output_folder_name):
                os.mkdir(self.output_folder_name)
            for idx, rect in enumerate(self.rects):
                x, y, w, h = rect
                crop_img = self.binary[y:y + h, x:x + w]
                # crop_img = self._remove_underline_preserve_text(gray=crop_img) # remove the cloze line

                # open and close
                horizon_k = 3 #int(math.sqrt(crop_img.shape[1])) 
                opening = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, (horizon_k, horizon_k))
                crop_img_ = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (horizon_k, horizon_k))
                crop_img_ = crop_img_.copy()
            
                self.crop_img.append(crop_img_)
                cv2.imwrite(os.path.join(self.output_folder_name, '{}-{}.jpg'.format(self.name, idx + 1)), crop_img_)
            return self.crop_img


    def _remove_underline_preserve_text(self, gray):
        
        detect_horizontal = self.__detectLines(gray)

        # Find contours of the lines
        contours = self.__contourExtraction(detect_horizontal)
        
        # Create a mask for areas to keep
        mask = np.ones(gray.shape[:2], dtype=np.uint8) * 255
        
        for contour in contours:
            x, y, w, h = contour
            margin = 1
            # Check for content above and below the line, column by column
            for col in range(x, x+w):
                
                above = gray[max(0, y-margin):y, col:col+1]
                below = gray[y+h:min(gray.shape[0], y+h+margin), col:col+1]
                
                # If there's no significant content above or below, mark this column for removal
                if np.sum(above) != 0 and np.sum(below) != 0:
                    mask[y:y+h, col] = 0
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Fill removed areas with white
        result[mask == 0] = 255
        
        # # Save the result
        # cv2.imwrite(output_img_path, result)
        
        return result

    @staticmethod
    def __cmp_rect(a, b):
        if (abs(a[1] - b[1]) < 10 and a[0] > b[0]) or a[1] > b[1]:
            return 1
        elif abs(a[1] - b[1]) < 10 and abs(a[0] - b[0]) < 20:
            return 0
        else:
            return -1

    @staticmethod
    def __cmp_rect_r(a, b):
        if (abs(a[1] - b[1]) < 5 and a[0] < b[0]) or a[1] > b[1]:
            return -1
        elif abs(a[1] - b[1]) < 5 and abs(a[0] - b[0]) < 5:
            return 0
        else:
            return 1


if __name__ == '__main__':
    # path = input('输入识别的文件夹路径\n')
    # debug = input('是否为Debug模式：T/F\n')
    # if debug == 'T':
    #     debug = True
    # elif debug == 'F':
    #     debug = False
    # else:
    #     print('输入错误')
    #     sys.exit()
    debug = False
    path = r'D:\code\AutoScore\AutoScore\OCRAutoScore\segmentation\blankSegmentation\img\demo.jpg'  # 文件夹名
    count = 0
    model = Model(debug=debug)
    res = model.process(path, os.path.basename(path))  # res存储分割的图片
