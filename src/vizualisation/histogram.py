import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

from utils import HOUSES, COLOR

df = pd.read_csv('../../datasets/dataset_train.csv', index_col=0)
features = df.columns[5:].to_list()
df = df.drop(df.iloc[:, 1:5], axis=1)

df_houses = {
    "Gryffindor": df[df["Hogwarts House"] == "Gryffindor"],
    "Slytherin": df[df["Hogwarts House"] == "Slytherin"],
    "Ravenclaw": df[df["Hogwarts House"] == "Ravenclaw"],
    "Hufflepuff": df[df["Hogwarts House"] == "Hufflepuff"]
}

fig = plt.figure(figsize=(12, 9))
axs = fig.subplots(nrows=3, ncols=5)
for i, feature in enumerate(features):
    for h in HOUSES:
        axs[i // 5][i % 5].hist(df_houses[h][feature].dropna(), bins=20, color=COLOR[h], label=h, alpha=0.5)
    axs[i // 5][i % 5].set_title(feature)
    axs[i // 5][i % 5].set_xlabel("Value")
    axs[i // 5][i % 5].set_ylabel("Frequency")

# Remove empty subplots
axs[2, 3].axis('off')
axs[2, 4].axis('off')

# Put margins between subplots
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Add legend
handles = [Rectangle((0,0),3,3,color=c,ec="k") for c in COLOR.values()]
plt.legend(handles, HOUSES)

plt.show()