import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()

# change the file destination to your data csv file here
data_file = "/home/hdo/ukbb_analyzer/Data/hieu_data_imputed_dominant_model_all.csv"
# change the name of the output figure here
figure = os.path.join(cwd, "data.png")
# plot gender against depression
data = pd.read_csv(data_file, usecols=["Sex", "PHQ9_binary"])

counts = data.groupby(['Sex', 'PHQ9_binary']).size().unstack()

# Plot the stacked bar chart
counts.plot(kind='bar', stacked=True)

# Add a legend
plt.legend(["Not Depressed", "Depressed"])

# Add x and y labels and a title
# plt.xlabel('Sex')
plt.xticks([0, 1], ['Female', 'Male'], rotation=0)
plt.ylabel('Count')
plt.title('Distribution of Data by Gender and Depression')

# Add text labels to the plot
for i in range(len(counts)):
    sum = 0
    for j in range(len(counts.columns)):
        count = counts.iloc[i, j]
        plt.text(i, sum + count/2,
                 str(count), ha='center', va='center')
        sum = sum + count

plt.savefig(figure)
