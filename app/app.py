import streamlit as st



#################
#---button_event---#
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "back_login" not in st.session_state:
    st.session_state.back_login = False

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

#---other---#
if "video" not in st.session_state:
    st.session_state.video = ''
    
#################
login_page = st.Page(r"pages\main.py", title="Healthcare", icon=":material/login:")
upload_page = st.Page(r"pages\upload.py", title="Healthcare-upload", icon=":material/upload:")
predict_page = st.Page(r"pages\predict.py", title="Healthcare-predict", icon='📈')

# pg = st.navigation([login_page]) #,upload_page,predict_page
# st.set_page_config(page_title="Data manager", page_icon=":material/edit:")

if st.session_state.logged_in:
    pg = st.navigation([upload_page])
else:
    pg = st.navigation([login_page]) 

# return to login
if st.session_state.back_login:
    pg = st.navigation([login_page])
    st.session_state.back_login = False

# go to predict
if st.session_state.uploaded:
    pg = st.navigation([predict_page])
    st.session_state.uploaded = False
    
pg.run()


