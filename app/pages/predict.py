import streamlit as st
import extraction.mp_predict as ext_pred
import tempfile
import time
import os
import pandas as pd

st.header('Prediction')

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(st.session_state.video.read())
        temp_filename = temp_file.name

video_pred, dataframe_pred = ext_pred.pred_in_video(temp_filename)
# if video_pred is not None:

if os.path.exists(video_pred):
    st.video(video_pred)
    st.dataframe(dataframe_pred)
    # st.markdown(f"[Open Video]({video_pred})")
else:
    st.error("Video file not found!")



# time.sleep(10)
# if video_pred:
#     video_file = open(f"{video_pred}", "rb")
#     video_bytes = video_file.read()

#     st.video(video_bytes)


# st.text(txt)

# progress_text = "Prediction in progress. Please wait."
# my_bar = st.progress(0, text=progress_text)

# # for percent_complete in range(100):
# percent_complete = 0
# flag_bar = True
# while flag_bar:
#     time.sleep(0.01)
#     percent_complete += 1
#     my_bar.progress(percent_complete, text=progress_text)


# time.sleep(1)
# my_bar.empty()