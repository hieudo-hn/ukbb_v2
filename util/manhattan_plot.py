# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def chromMap():
    file = "/home/hdo/genes/allSNPs.txt"
    map = {}

    f = open(file, 'r')
    line = f.readline()
    while line and len(line) > 0:
        info = line.split(', ')
        chrom = int(info[0])
        snp = info[1].strip()
        map[snp] = chrom
        line = f.readline()
    f.close()

    return map


chrom_map = chromMap()

# change this
chi2_data_file = '/home/hdo/ukbb_analyzer/Data/hieu_chi2Data_Male.csv'
label_cut_off = 8                                                 
plot_title = "Manhattan Plot of SNPs After Chi-square Feature Selection for Male"
figure_file = 'manhattan_plot_male.png'

# sample data
df = pd.read_csv(chi2_data_file)

# -log_10(pvalue)
df['minuslog10pvalue'] = -np.log10(df['p_val'])
df['chromosome'] = [chrom_map[snp] if snp in chrom_map.keys() else -
                    1 for snp in list(df['SNPs'])]
df = df.sort_values('chromosome')

# How to plot gene vs. -log10(pvalue) and colour it by chromosome?
df['ind'] = range(len(df))
df_grouped = df.groupby(('chromosome'))

# manhattan plot
fig = plt.figure(figsize=(14, 8))  # Set the figure size
ax = fig.add_subplot(111)
colors = ['darkred', 'darkgreen', 'darkblue', 'gold', 'red']
x_labels = []
x_labels_pos = []
for num, (name, group) in enumerate(df_grouped):
    if (int(name) < 0):
        continue
    group.plot(kind='scatter', x='ind', y='minuslog10pvalue',
               color=colors[num % len(colors)], ax=ax)
    x_labels.append(name)
    pos = group['ind'].iloc[-1] - \
        (group['ind'].iloc[-1] - group['ind'].iloc[0])/2
    x_labels_pos.append(pos)
    # label
    to_label = group[group['minuslog10pvalue'] > label_cut_off]
    prev_y, change = 0, 0
    for i, row in to_label.iterrows():
        # avoid overlapping label
        cur_y = row['minuslog10pvalue']
        if (cur_y == prev_y):
            change = change + 0.2
            cur_y = cur_y + change
        else:
            change = 0
        ax.annotate(row['SNPs'], xy=(pos, cur_y))
        prev_y = row['minuslog10pvalue']
ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels)

# set axis limits
ax.set_xlim([0, len(df)])
ax.set_ylim([0, math.ceil(df['minuslog10pvalue'].max())])

# x axis label
ax.set_xlabel('Chromosome')
ax.set_ylabel('$-log_{10}(p_{value})$')
ax.set_title(plot_title)

# show the graph
plt.savefig(figure_file)
