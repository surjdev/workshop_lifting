import streamlit as st
st.header('Login')

if st.button('login'):#go to upload
    st.session_state.logged_in = True
    st.rerun()