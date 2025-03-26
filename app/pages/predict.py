import streamlit as st
import extraction.mp_predict as ext_pred
import models.filter_tools as ft
import models.predict_angle as pa
import models.predict_angle as test

import tempfile
import time
import os
import pandas as pd
import cv2
import numpy as np
import re

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

st.header('Prediction')

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(st.session_state.video.read())
        temp_filename = temp_file.name

dataframe_pred = ext_pred.pred_in_video(temp_filename)

kps = dataframe_pred.drop('time',axis=1)
time_ = dataframe_pred["time"]

data = kps.values
n_frames, n_columns = data.shape
n_keypoints = n_columns // 3 

data_reshaped = data.reshape(n_frames, n_keypoints, 3)

smoothed_data = ft.smooth_keypoints(data_reshaped, window_size=5)
smoothed_df = pd.DataFrame(smoothed_data.reshape(n_frames, n_columns), columns=kps.columns)

df_combined = pd.concat([time_, smoothed_df], axis=1)

# Determine keypoint indices based on column names (e.g., keypoint_0_x, keypoint_1_x, etc.)
keypoint_columns = [col for col in df_combined.columns if re.match(r'keypoint_\d+_x', col)]
keypoint_indices = sorted(set(int(re.search(r'keypoint_(\d+)_x', col).group(1)) for col in keypoint_columns))


col1, col2 = st.columns(2)
# col3, col4 = st.columns(2)

placeholder1 = st.empty()
placeholder2 = col1.empty()
# placeholder3 = col3.empty()
# placeholder4 = col4.empty()

if temp_filename:
    cap = cv2.VideoCapture(temp_filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction_df = pd.read_excel(r"D:\git_project\workshop_lifting\app\Train\dataset_test.csv_with_predictions.xlsx")

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        row = df_combined.iloc[(df_combined['time'] - current_time).abs().argsort()[:1]]
        pred_row = row

        label = pred_row.get('prediction', "No Prediction")
        if isinstance(label, pd.Series):
            label = label.iloc[0]

        time_val = pred_row.get('time', current_time)
        if isinstance(time_val, pd.Series):
            time_val = time_val.iloc[0]

        text = f"Time: {time_val:.2f}s | {label}"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1.5  # Increase font size
        # thickness = 2
        # position = (50, 50)
        # (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # x, y = position

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f = frame_rgb.copy()

        for kp in keypoint_indices:
            # Extract x, y coordinates and visibility for the keypoint
            x = row[f'keypoint_{kp}_x'].values[0]
            y = row[f'keypoint_{kp}_y'].values[0]
            # z = row[f'keypoint_{kp}_z'].values[0]
            # vis = row[f'keypoint_{kp}_visibility'].values[0]

            x_px = int(x * frame_rgb.shape[1])
            y_px = int(y * frame_rgb.shape[0])
            
            # Draw a circle and label the keypoint index
            cv2.circle(frame_rgb, (x_px, y_px), 5, (0, 255, 0), -1)
            cv2.putText(frame_rgb, str(kp), (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        draw_text(frame_rgb, text, font_scale=4, pos=(50,50), text_color=(0, 0, 0), text_color_bg=(255, 255, 255))

        # สร้างภาพจำลองจากฟังก์ชัน draw_frame_from_keypoints()
        frame_img = pa.draw_3d_pose_from_keypoints(df_combined, frame=row.iloc[0])

        # แสดงภาพในสองคอลัมน์: ซ้ายภาพปกติ, ขวาภาพจากฟังก์ชัน
        placeholder1.image(frame_rgb, caption="ภาพปกติ", use_container_width=True)
        placeholder2.image(frame_img, caption="กราฟจำลองการวัดมุม", use_container_width=True)

        # placeholder3.image(frame_img, caption="กราฟจำลองการวัดมุม", use_container_width=True)
        # i += 1

    cap.release()
    # เคลียร์พื้นที่แสดงผลเมื่อจบการแสดงวิดีโอ
    # placeholder1.empty()
    # placeholder2.empty()