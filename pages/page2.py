import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Data Visualization")

#Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)
#Plot Data
fig, ax = plt.subplots()
ax.plot(x,y)
#Display the plot
st.pyplot(fig)

