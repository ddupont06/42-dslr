import matplotlib.pyplot as plt
import pandas as pd

from utils import COLOR

df = pd.read_csv('../../datasets/dataset_train.csv')
df.dropna(axis=0, inplace=True)
df_houses = {
    "Gryffindor": df[df["Hogwarts House"] == "Gryffindor"],
    "Slytherin": df[df["Hogwarts House"] == "Slytherin"],
    "Ravenclaw": df[df["Hogwarts House"] == "Ravenclaw"],
    "Hufflepuff": df[df["Hogwarts House"] == "Hufflepuff"]
}

fig = plt.figure(figsize=(12, 9))
axs = fig.subplots(nrows=2, ncols=2)

for house, ax in zip(df_houses.items(), axs.flatten()):
    name, data = house
    ax.scatter(data["Defense Against the Dark Arts"], data["Astronomy"], color=COLOR[name], label=name, alpha=0.5)
    ax.set_title(name)
    ax.set_xlabel("Defense Against the Dark Arts")
    ax.set_ylabel("Astronomy")

# Put margins between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()


