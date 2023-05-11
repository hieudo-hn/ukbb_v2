import os

import pandas as pd

# this file parse all SNPs and their chromosome number from csv to a text file
cwd = os.getcwd()
geneFolder = os.path.join(cwd, "genes")
outfile = os.path.join(geneFolder, "allSNPs.txt")

res = []
for file in os.listdir(geneFolder):
    if file.endswith("csv"):
        filepath = os.path.join(geneFolder, file)
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            snp = row["name"]
            chrom = row["#\"chrom\""]
            if chrom.endswith("alt"):
                chrom = "chr11"
            chrom = int(chrom[3:])
            res.append("{}, {}\n".format(chrom, snp))
res.sort()

f = open(outfile, 'w')
for n in res:
    f.write(n)
f.close()
