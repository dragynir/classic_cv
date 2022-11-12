import cv2
import matplotlib.pyplot as plt
import streamlit as st

# коррекция яркости и контраста
#
# def load_gray_image(path):
#     return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
#

st.checkbox('yes')
st.button('Click')
st.radio('Pick your gender',['Male','Female'])
st.selectbox('Pick your gender',['Male','Female'])
st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0,50)

st.write("Hello ,let's learn how to build a streamlit app together")
