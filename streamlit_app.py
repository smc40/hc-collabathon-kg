import streamlit as st
import pandas as pd

from src.utils import DATA_FILE

with open(DATA_FILE) as dfile:
    df = pd.read_csv(dfile)

st.title('Bob Blobs')

