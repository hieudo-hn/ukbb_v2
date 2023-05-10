library(car)
library(dplyr)

# this csv file should only have SNPs and the last column is your target 
# variable (hard-coded - PHQ9_binary, you should change the code down here)
data = read.csv("~/Desktop/male.csv")
cols = colnames(data)[2:(length(colnames(data))-1)] # remove the index column
mdl <- glm(as.formula(paste("PHQ9_binary~", paste(cols, collapse="+"))), data=
             data, family=binomial)
a = as.data.frame(alias(mdl)$Complete)
to_remove = rownames(a)
after_remove = setdiff(cols, to_remove)
mdl <- glm(as.formula(paste("PHQ9_binary~", paste(after_remove, collapse="+"))), data=
             data, family=binomial)
gvif_df = as.data.frame(vif(mdl)) 
gvif_df = gvif_df %>% arrange(vif(mdl))
write.csv(gvif_df, file="~/Desktop/gvif_male.csv")