import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def auto_crop(image_path):
    # image_path = image_path.split('\n')[0]
    image = cv2.imread(image_path)
    orig = image.copy()

    # 获取图片尺寸并计算动态参数
    height, width = image.shape[:2]
    size_factor = max(height, width) / 1000  # 基于1000px为基准的比例因子

    # 图像增强参数根据尺寸调整
    alpha = 1.5 + 0.3 * size_factor  # 动态对比度
    beta = 30 * size_factor         # 动态亮度
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 自适应直方图均衡化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 动态模糊核大小 (奇数)
    blur_size = max(3, int(5 * size_factor)) | 1  # 确保为奇数
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # 动态Canny阈值
    edged = cv2.Canny(gray, int(50 * size_factor), int(150 * size_factor))
    # 动态形态学操作核大小
    kernel_size = max(3, int(5 * size_factor))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edged,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=int(100 * size_factor),
                            minLineLength=int(100 * size_factor),
                            maxLineGap=int(10 * size_factor))
    if lines is None:
        return None, "No lines detected"
    lines = pd.DataFrame(lines.reshape(-1,4))
    lines['length'] = np.sqrt((lines[2]-lines[0])**2 + (lines[3]-lines[1])**2)
    lines['x'] = lines[0]+lines[2]
    lines['y'] = lines[1]+lines[3]
    lines['angle'] = abs(np.arctan2(lines[3]-lines[1], lines[2]-lines[0])*180/np.pi)
    # 横着的线
    lines_horizontal = lines[lines['angle'].between(0, 40)]
    # 筛选长度大于最大线1/2的直线
    max_length = lines_horizontal['length'].max()
    lines_horizontal = lines_horizontal[lines_horizontal['length'] > max_length/2]
    # 取y最大和最小的两条直线
    topmost = lines_horizontal.loc[lines_horizontal['y'].idxmin()][[0,1,2,3]].tolist()
    bottommost = lines_horizontal.loc[lines_horizontal['y'].idxmax()][[0,1,2,3]].tolist()
    # 竖着的线
    lines_vertical = lines[lines['angle'].between(50,90)]
    # 筛选长度大于最大线1/2的直线
    max_length = lines_vertical['length'].max()
    lines_vertical = lines_vertical[lines_vertical['length'] > max_length/2]
    # 取x最大和最小的两条直线
    leftmost = lines_vertical.loc[lines_vertical['x'].idxmin()][[0,1,2,3]].tolist()
    rightmost = lines_vertical.loc[lines_vertical['x'].idxmax()][[0,1,2,3]].tolist()
    # 绘制四条边界线
    # plt.figure()
    # for lien in lines.values:
    #     x1, y1, x2, y2 = lien[:4]
    #     plt.plot([x1, x2], [y1, y2], color='blue')
    # for line in [topmost, bottommost, leftmost, rightmost]:
    #     x1, y1, x2, y2 = line
    #     plt.plot([x1, x2], [y1, y2], color='red')
    # plt.show()
    # 计算四条边界线的交点
    def line_intersection(line1, line2):
        # 解包直线1的两个点 (x1, y1), (x2, y2)
        x1, y1, x2, y2 = line1
        # 解包直线2的两个点 (x3, y3), (x4, y4)
        x3, y3, x4, y4 = line2

        # 计算分母
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        # 如果分母为0，表示两直线平行或重合
        if denom == 0:
            return None

        # 计算ua, ub
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        # 计算交点坐标
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)

        return [int(x), int(y)]

    # 计算四个角点
    try:
        tl = line_intersection(leftmost, topmost )
        tr = line_intersection(topmost, rightmost)
        br = line_intersection(rightmost, bottommost)
        bl = line_intersection(bottommost, leftmost)
    except:
        return None, '无法计算有效交点'

    # 组成四边形坐标
    screenCnt = np.array([tl, tr, br, bl], dtype=np.float32)
    # 使用matplotlib绘制四条边界线
    # plt.figure()
    # for i in range(4):
    #     plt.plot([screenCnt[i-1][0], screenCnt[i][0]],[screenCnt[i-1][1], screenCnt[i][1]])
    # plt.show()
    if screenCnt is None:
        print("请上传清晰的电力铭牌照片")
        return None,'请上传清晰的电力铭牌照片'
    warped = four_point_transform(orig, screenCnt.reshape(4, 2))
    cv2.imshow("Warped", warped)
    return warped,'图片裁剪成功'

if __name__ == '__main__':
    # 使用示例
    os.chdir('../datasets/DLMP/')
    image_path = './org_imgs/'
    if image_path.endswith('/'):
        for img_name in os.listdir(image_path):
            if not img_name.endswith('.jpg'):
                continue
            img_path = image_path + img_name
            cropped_image,info = auto_crop(img_path)
            print(img_name,info)
            if cropped_image is not None:
                # print('./crop/org_imgs/' + img_name)
                cv2.imwrite('./crop/org_imgs/' + img_name, cropped_image)
    else:
        cropped_image,_ = auto_crop(image_path)
        if cropped_image is not None:
            # cv2.imshow("Cropped Image", cropped_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # 保存crop
            cv2.imwrite(f'./crop/{image_path.split("/")[-1]}', cropped_image)
