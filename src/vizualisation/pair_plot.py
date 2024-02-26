import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import COLOR

df = pd.read_csv('../../datasets/dataset_train.csv', index_col=0)
df.dropna(axis=0, inplace=True)
df = df.drop(df.iloc[:, 1:5], axis=1)

sns.set(font_scale=0.5)
sns.set_style("ticks")
sns.pairplot(df, hue='Hogwarts House', palette=COLOR, height=0.6, aspect=1.5, plot_kws={'s': 1})

plt.show()
