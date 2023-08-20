import pandas as pd
import numpy as np
from scipy.io import savemat
import os
import cv2
import re




def process_csv_files(folder_path):

    files = os.listdir(folder_path)

    for file_name in files:


        if file_name.endswith('.csv'):

            file_path = os.path.join(folder_path, file_name)
            print(file_path)
            match = re.search(r"from(\d+)_", file_name)
            print(match)
            k = match.group(1)
            image_path1 = folder_path +k +'l' +'.png'
            image_path2 = folder_path +k +'r'+ '.png'

            image1 = cv2.imread(image_path1)
            image2 = cv2.imread(image_path2)
            # cv2.imshow("",image1)

            height1, width1 = image1.shape[:2]
            height2, width2 = image2.shape[:2]
            # print(height1,width1,height2,width2)
            #
            hmax= max(height1,height2)
            wmax = max(width1,width2)
            # hmax= 640
            # wmax = 480

            #
            merged_width = width1 + width2
            merged_height = max(height1, height2)


            merged_image = np.zeros((merged_height, merged_width, 3), dtype=np.uint8)

            merged_image[:height1, :width1] = image1
            merged_image[:height2, width1:] = image2

            whiteimage=np.zeros((hmax, wmax, 3), dtype=np.uint8)


            # 读取CSV文件
            data = pd.read_csv(file_path)
            extracted_data = []
            x = []
            y = []
            CorrectIndex = []


            for i in range(0, len(data), 31):

                values = data.iloc[i, [0, 3, 4, 5, 6]].values

                x.append([values[1], values[2]])
                y.append([values[3], values[4]])


                if values[0] == 1:
                    CorrectIndex.append(i // 31 +1)
                    pt1 =     (int(values[1]*wmax), int(values[2]*hmax))
                    pt2 =     (int(values[3]*wmax)+width1, int(values[4]*hmax))
                    pt2_ =     (int(values[3]*wmax), int(values[4]*hmax))
                    cv2.line(merged_image, pt1, pt2, (0, 255, 0), 2)
                    cv2.line(whiteimage, pt1, pt2_, (0, 255, 0), 2)
            x_data = np.array(x)
            y_data = np.array(y)
            CorrectIndex_data = np.array(CorrectIndex).reshape(-1, 1)
            CorrectIndex_data = CorrectIndex_data.astype(np.float64)


            cv2.imshow('Merged Image', merged_image)
            cv2.imshow('whiteimage', whiteimage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            x_data[:,0] *= wmax
            x_data[:,1] *= hmax
            y_data[:,0] *=wmax
            y_data[:,1] *=hmax

            mat_dict = {
                'X': x_data,
                'Y': y_data,
                'CorrectIndex': CorrectIndex_data
            }


            mat_file_name = os.path.splitext(file_name)[0] + '.mat'
            mat_file_path = os.path.join(folder_path, mat_file_name)
            savemat(mat_file_path, mat_dict)



# folder_path = r'./split low/out/'
folder_path = r'./hanyang2002/'

process_csv_files(folder_path)
