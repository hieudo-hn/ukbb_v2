# Cluster Setup:
- To run on cluster, first connect to Colgate VPN, after that open terminal and ssh into turing using your turing account.
Eg. if using Colgate turing account, type:
```ssh username@turing.colgate.edu```
- After that, type in your turing password
- Download Anaconda installer onto your laptop from here: https://www.anaconda.com/download/#linux
- Upload the downloaded file to the cluster using either Cyberduck or ssh or Vscode (whetever is convenient for you)
- Next, create a new conda environment with Python and R:
```Conda create --name <name-of-environment> python=3.10 R=4.2```
- How to activate the environment:
```conda activate <name-of-environment>```
- Drive Link: 
```https://drive.google.com/drive/folders/11WjLJ99bSir_o1pJJFBG1XdoAt_8DQyM?usp=sharing```

# Prerequisite:

- For more information on how to access UKBB data, take a look at this link:
https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/ukbgene_instruct.html
- Your key file is in the key folder

1. Installing gfetch:
```
wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/gfetch
chmod 755 .gfetch
```

2. Install ukbgene:
```
wget  -nd  biobank.ndph.ox.ac.uk/ukb/ukb/util/ukbgene_linkset.tar
tar -xvf ukbgene_linkset.tar
make ukbgene
```

3. Install ukbb_parser:
The source code: https://github.com/nadavbra/ukbb_parser
```
git clone https://github.com/nadavbra/ukbb_parser.git /tmp/ukbb_parser_src
cd /tmp/ukbb_parser_src
git submodule update --init --recursive
python setup.py install
```

You need to move the `.ukbb_paths.py` to the root directory as well as the Data folder. Modify the configurations in `.ukbb_paths.py` per instructions.

4. Move ukb52255.csv to the Data folder. This should be found in the google drive shared with you.

5. Install PLINK:
```
conda install -c conda-forge pandas-plink
wget  -nd  biobank.ndph.ox.ac.uk/ukb/ukb/auxdata/ukb_snp_bim.tar
tar -xvf ukb_snp_bim.tar 
```
Move output to genetics folder in Data

6. Install python dependencies:
```
pip install -r requirements.txt
```
In case we miss some dependencies, you can manually pip install it.

# Getting UKBB raw data:
For more information, check section 4 and 5 of https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/ukbgene_instruct.html.
- Note: if you ever encounter symbolic link issue, there is a file called symbolic_link.py that can help you with it.
* Example usage:
- to get .bed file for chromosome 17
```
./ukbgene -a k85474r52255.key -c17
```
- to get .fam file for chromosome 17
```
./ukbgene -a k85474r52255.key -c17 -m
```
- to get .bim file follow the intructions here: https://biobank.ctsu.ox.ac.uk/crystal/refer.cgi?id=1963

# Using this repo:
- FeatureSelection.py: has code for chi-square feature selection and how to use it
- MFNN.py: definition of multilayer feedforward neural network class, has functions to intialize, train & evaluate and plot results
- MachineLearning.py: has code for other ML classifiers like naive_bayes, xgboost, random forest
- main.py: where most of your code goes, has codes for multivariate logistic regression

