#uncomment this if you need to install packages
#install.packages("SNPassoc")
#install.packages("dplyr")
library("SNPassoc")
library(dplyr)

#################################################################

# change to your desired SNP here
female = c("rs79590198","rs78929565","rs1398731","rs12484542",
         "rs74823498","rs76844436","rs885747","rs79495223",
         "rs6825994","rs12578274","rs35488012")
male = c("rs138102314","rs117516155","rs74020725","rs114825723",
         "rs76379455")
overall = c("rs78847165", "rs35488012", "rs885747")

# change clinical factor and target as you want
clinical = c("Age", "Chronotype", "Sleeplessness.Insomnia")
target = "PHQ9_binary"

# change the location of your data here
data = read.csv("~/Desktop/result.csv")
snp = colnames(data)[4:19] # these are the snp columns

#################################################################

data <- data %>%
  mutate_at(vars(4:19), 
            list(~ ifelse(. == 0, "AA", ifelse(. == 1, "Aa", .))))

# female
data.female <- setupSNP(data=data %>%
                     filter(Sex == 0) %>%
                     select(all_of(c(female, clinical, target))),
                   colSNPs=1:length(female), sep="")
result.female = interactionPval(as.formula(paste("PHQ9_binary~",
                                          paste(clinical, collapse="+"))), 
                         data.female)
write.csv(result.female, file = "~/Desktop/snp_snp_female.csv", row.names = TRUE)
plot(result.female)

# male
data.male <- setupSNP(data=data %>%
                          filter(Sex == 1) %>%
                          select(all_of(c(male, clinical, target))),
                        colSNPs=1:length(male), sep="")
result.male = interactionPval(as.formula(paste("PHQ9_binary~",
                                          paste(clinical, collapse="+"))), 
                         data.male)

write.csv(result.male, file = "~/Desktop/snp_snp_male.csv", row.names = TRUE)
plot(result.male)

# overall
clinical = c("Sex", clinical)
data.overall <- setupSNP(data=data %>%
                          select(all_of(c(overall, clinical, target))),
                        colSNPs=1:length(overall), sep="")
result.overall = interactionPval(as.formula(paste("PHQ9_binary~",
                                          paste(clinical, collapse="+"))), 
                         data.overall)
write.csv(result.overall, file = "~/Desktop/snp_snp_overall.csv", row.names = TRUE)
plot(result.overall)
