import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Sample data
data = {"Model": ["Model A", "Model B", "Model C"], "Points": [85, 75, 60]}
df = pd.DataFrame(data)

# Title
st.title("ML Model Championship Leaderboard")

# Leaderboard Table
st.table(df)

# Sidebar for navigation
model = st.sidebar.selectbox("Select a model", df["Model"])
st.write(f"Detail for {model}")

# Sample chart
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Points"])
st.pyplot(fig)
