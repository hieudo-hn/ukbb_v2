from skfeature.function.information_theoretical_based import LCSI
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import pandas as pd


def mrmr(X, y, **kwargs):
    """
    This function implements the MRMR feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if "n_selected_features" in kwargs.keys():
        n_selected_features = kwargs["n_selected_features"]
        F, J_CMI, MIfy = LCSI.lcsi(
            X, y, gamma=0, function_name="MRMR", n_selected_features=n_selected_features
        )
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name="MRMR")
    return F


def jmi(X, y, **kwargs):
    """
    This function implements the JMI feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if "n_selected_features" in kwargs.keys():
        n_selected_features = kwargs["n_selected_features"]
        F, J_CMI, MIfy = LCSI.lcsi(
            X, y, function_name="JMI", n_selected_features=n_selected_features
        )
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name="JMI")
    return F


def run_chi2(X, y, file):
    chi2_score, p_val = chi2(X.to_numpy(), y)

    df = pd.DataFrame()
    df["SNPs"] = X.columns
    df["chi2_score"] = chi2_score.tolist()
    # print(chi2_score)
    df["p_val"] = p_val.tolist()
    df.sort_values(by="p_val", inplace=True)
    df.to_csv(file)


def calc_odds_ratio(
    data, result_file="/home/hdo/ukbb_analyzer/Data/hieu_chi2Data_Final.csv"
):
    result = pd.read_csv(result_file, index_col=0)
    dd, od, dn, on, ors = [], [], [], [], []
    for snp in result["SNPs"]:
        if not snp.startswith("rs"):
            continue
        temp = data[[snp, "PHQ9_binary"]]
        dominant_depressed = temp.loc[
            (temp[snp] == 0) & (temp["PHQ9_binary"] == 1)
        ].shape[0]
        other_depressed = temp.loc[(temp[snp] != 0) & (temp["PHQ9_binary"] == 1)].shape[
            0
        ]
        dominant_notDepressed = temp.loc[
            (temp[snp] == 0) & (temp["PHQ9_binary"] == 0)
        ].shape[0]
        other_notDepressed = temp.loc[
            (temp[snp] != 0) & (temp["PHQ9_binary"] == 0)
        ].shape[0]
        odds_ratio = ((other_depressed + 0.5) * (dominant_notDepressed + 0.5)) / (
            (dominant_depressed + 0.5) * (other_notDepressed + 0.5)
        )
        dd.append(dominant_depressed)
        od.append(other_depressed)
        dn.append(dominant_notDepressed)
        on.append(other_notDepressed)
        ors.append(odds_ratio)

    result["Dominant Depressed"] = dd
    result["Dominant Not Depressed"] = dn
    result["Other Depressed"] = od
    result["Other Not Depressed"] = on
    result["Odds Ratio"] = ors
    result.to_csv(result_file)


def benjamini_hochberg_correction(
    result_file="/home/hdo/ukbb_analyzer/Data/hieu_chi2Data_Final.csv",
):
    result = pd.read_csv(result_file, index_col=0)
    drop(result, "p_val")
    adjusted = fdrcorrection(result["p_val"], method="indep", is_sorted=False)[1]
    result["B-H p-val"] = adjusted
    result.to_csv(result_file)


def drop(data, column):
    col = np.where(data.columns == column)[0][0]
    to_drop = []
    for i in range(data.shape[0]):
        if np.isnan(data.iloc[i, col]) or data.iloc[i, col] < 0:
            to_drop.append(i)
    data.drop(data.index[to_drop], inplace=True)


def load_snp():
    result = []
    with open("/home/hdo/genes/7ksnps.txt", "r") as f:
        for line in f:
            result.append(line.strip())
    return result


################# chi-2 feature selection #########################

file = "/home/hdo/ukbb_analyzer/Data/hieu_data_imputed_dominant_model_all.csv"
target = "PHQ9_binary"
chi2_result_file = "/home/hdo/ukbb_analyzer/Data/hieu_chi2Data_Overall.csv"
snp = load_snp()
snp.append(target)
# snp.append("Sex")
data = pd.read_csv(file, usecols=snp)
# data = data[data["Sex"] == 0]

y_1 = data["PHQ9_binary"].to_numpy()
x_1 = data.drop(["PHQ9_binary"], axis=1)

run_chi2(x_1, y_1, chi2_result_file)
calc_odds_ratio(data, chi2_result_file)
benjamini_hochberg_correction(chi2_result_file)

# if you want to draw manhattan_plot there is a file in util folder
