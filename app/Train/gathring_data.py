import sys
sys.path.append(r'D:\git_project\workshop_lifting\app')

import extraction.mp_predict as ext_pred
import pandas as pd


temp_filename = r"D:\Dataset\AI_W2_healthcare\lifting_0215.mp4"
video_pred, dataframe_pred = ext_pred.pred_in_video(temp_filename)
print(video_pred)
# print(dataframe_pred)
dataframe_pred.to_csv("dataset_test.csv",index=False)