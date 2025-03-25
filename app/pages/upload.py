import streamlit as st
import tempfile

#----page-setup----#
st.header('Upload Video')
uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.session_state.video = uploaded_file
    # st.text('Video Uploaded')
    if st.button('Click to predict',key='predict'):
        st.session_state.uploaded = True
        st.rerun()

if st.button('back'):
    st.session_state.back_login = True
    st.rerun()
