from skfeature.function.information_theoretical_based import LCSI
import statsmodels.api as sm

from statsmodels.stats.multitest import fdrcorrection

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from itertools import combinations
from MFNN import MFNN


# load snp from text file
def load_snp(file="/home/hdo/genes/7ksnps.txt"):
    result = []
    with open(file, "r") as f:
        for line in f:
            result.append(line.strip())
    return result


# modify the specified SNP in the snp array into dominant model
def dominant_model(data, snp):
    temp = data[snp]
    data[snp] = [0 if i == 0 else 1 for i in temp]


# multivarite logistic regression
def mvlog(Y_train, x_train, result_file):
    X_train_sm = sm.add_constant(x_train)
    logm2 = sm.GLM(Y_train, X_train_sm, family=sm.families.Binomial())
    res = logm2.fit()
    display_result(X_train_sm.columns, res, result_file)


# multivarite linear regression
def mvlinear(Y_train, x_train, result_file):
    X_train_sm = sm.add_constant(x_train)
    logm2 = sm.GLM(Y_train, X_train_sm, family=sm.families.Gaussian())
    res = logm2.fit()
    display_result(X_train_sm.columns, res, result_file)


# multinomial logistic regression
def multinomial_mvl(Y_train, x_train, result_file):
    X_train_sm = sm.add_constant(x_train)
    multinomial_log = sm.MNLogit(Y_train, X_train_sm)
    res = multinomial_log.fit()
    display_result(X_train_sm.columns, res, result_file)


def display_result(variable, res, result_file):
    print(res.summary())
    df = pd.DataFrame(variable, columns=["variable"])
    df = pd.merge(
        df,
        pd.DataFrame(res.params, columns=["coefficient"]),
        left_on="variable",
        right_index=True,
    )
    conf_int = pd.DataFrame(res.conf_int())
    conf_int = conf_int.rename({0: "2.5%", 1: "97.5%"}, axis=1)
    df = pd.merge(df, conf_int, left_on="variable", right_index=True)
    df = pd.merge(
        df,
        pd.DataFrame(res.bse, columns=["std error"]),
        left_on="variable",
        right_index=True,
    )
    df = pd.merge(
        df,
        pd.DataFrame(res.pvalues, columns=["pvalues"]),
        left_on="variable",
        right_index=True,
    )
    adjusted = fdrcorrection(res.pvalues, method="indep", is_sorted=False)[1]
    df["adjusted pval"] = adjusted
    df = df.sort_values(by="adjusted pval")
    df.to_csv(result_file)

    print("AIC: {}".format(res.aic))
    print("BIC: {}".format(res.bic))


# make pairwise for epistatic relationship
def makePairwise(df, columns_to_merge):
    pairs = list(combinations(columns_to_merge, 2))
    ohc = []
    for pair in pairs:
        # Create new column name as combination of original column names
        new_col_name = f"{pair[0]}_{pair[1]}"
        df[new_col_name] = df[pair[0]].astype(str) + df[pair[1]].astype(str)
        ohc.append(new_col_name)
    return pd.get_dummies(df, columns=ohc, drop_first=True)


# snp = load_snp()
literature_snp = load_snp("/home/hdo/genes/26SNPs.txt")
cutoff = 10

clinical_factor = ["Sex", "Age", "Chronotype", "Sleeplessness/Insomnia"]
target = "PHQ9_binary"

chi2_fs_result_file = "/home/hdo/ukbb_analyzer/Data/hieu_chi2Data_Male.csv"
mrmr_fs_result_file = "/home/hdo/ukbb_analyzer/Data/MRMRFeatures_Female.csv"
data_file = "/home/hdo/ukbb_analyzer/Data/hieu_data_imputed_dominant_model_all.csv"
gvif_result = "/home/hdo/ukbb_analyzer/Data/gvif_male.csv"

########### LOADING CHI2 RESULT ###############
# fs_result = pd.read_csv(chi2_fs_result_file, nrows=200)
# cols = fs_result["SNPs"].tolist()
# just_snp = pd.read_csv(data_file, usecols=cols)
# cols.append(target)
# cols.append("Sex")
# cols = cols + clinical_factor

####################### JMI MRMR Feature Selection ########################
# data = pd.read_csv(data_file, usecols=cols)
# data = data[data['Sex'] == 0]
# X = data.drop([target, 'Sex'], axis=1)
# y = data[target]
# F, J_CMI, MIfy = LCSI.lcsi(np.array(X), np.array(y),
#                            gamma=0, function_name='MRMR')
# res = X.iloc[:, F]
# res.to_csv(mrmr_fs_result_file)

################################################################
# X_train = data.drop([target], axis=1)
# just_snp = X_train.drop(clinical_factor, axis=1)

# # dominant model
# df = 1
# for col in just_snp.columns:
#     temp = X_train[col]
#     X_train[col] = [0 if i == 0 else 1 for i in temp]

########### LOADING MRMR RESULT ###############
# mrmr_result = pd.read_csv(mrmr_fs_result_file, nrows=1, index_col=0)
# snp = mrmr_result.columns.tolist()
# cols = snp + [target, 'Sex']
# data = pd.read_csv(data_file, usecols=cols)
# data = data[data['Sex'] == 1]
# data.drop(['Sex'], axis=1, inplace=True)
# data.to_csv("male.csv")


######### Loading GVIF result ###############
# gvif_df = pd.read_csv(gvif_result, index_col=0)

# # retain
# snp = gvif_df[gvif_df['vif(mdl)'] < cutoff].index.tolist()
# cols = snp + [target]
# cols = cols + clinical_factor
# data = pd.read_csv(data_file, usecols=cols)
# data = data[data['Sex'] == 1]
# data.drop(['Sex'], axis=1, inplace=True)

# X = data.drop([target], axis=1)
# y = data[target]

# # OHC
# X_train = pd.get_dummies(X, columns=["Chronotype",
#                                      "Sleeplessness/Insomnia"], drop_first=True)

# mvlog(y, X_train, "/home/hdo/male.csv")

################### snp-snp #####################
# female
# snp = ["rs79590198", "rs78929565", "rs1398731", "rs12484542",
#       "rs76844436", "rs885747", "rs12578274", "rs6825994"]
# male
snp = ["rs138102314", "rs117516155", "rs74020725", "rs114825723", "rs76379455"]
to_pairwise = ["rs114825723", "rs117516155"]
# overall
# snp = ["rs885747", "rs35488012", "rs78847165"]
cols = snp + [target]
cols = cols + clinical_factor

data = pd.read_csv(data_file, usecols=cols)
data = data[data["Sex"] == 1]
data.drop(["Sex"], axis=1, inplace=True)
# OHC
data = pd.get_dummies(
    data, columns=["Chronotype", "Sleeplessness/Insomnia"], drop_first=True
)

##################### DEEP LEARNING #########################

data = makePairwise(data, to_pairwise)
mfnn_config = "/home/hdo/ukbb_analyzer/MFNN_config/male.json"
mfnn = MFNN(mfnn_config, data, target)
mfnn.train()
