######################################################################################
# JAX Internship R-Script
# Modeling
# written by Kayla Dixon

setwd("/personal/ksdixo23/JAXInternshipStuff/sample25/")

#setting up sample25 dataframe
B6J3 <- read.csv(file=file.choose())
rownames(B6J3) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                     "Segmentation_Average", "Segmentation_Median")
colnames(B6J3) <-c("B6J3")
B6J3 <- as.data.frame(t(B6J3))
B6JFPSY <- read.csv(file=file.choose())
rownames(B6JFPSY) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                    "Segmentation_Average", "Segmentation_Median")
colnames(B6JFPSY) <-c("B6JFPSY")
B6JFPSY <- as.data.frame(t(B6JFPSY))
B6JMPSY <- read.csv(file=file.choose())
rownames(B6JMPSY) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                       "Segmentation_Average", "Segmentation_Median")
colnames(B6JMPSY) <-c("B6JMPSY")
B6JMPSY <- as.data.frame(t(B6JMPSY))
B6NFPSY1500 <- read.csv(file=file.choose())
rownames(B6NFPSY1500) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                       "Segmentation_Average", "Segmentation_Median")
colnames(B6NFPSY1500) <-c("B6NFPSY1500")
B6NFPSY1500 <- as.data.frame(t(B6NFPSY1500))
B6NFPSY1571 <- read.csv(file=file.choose())
rownames(B6NFPSY1571) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                           "Segmentation_Average", "Segmentation_Median")
colnames(B6NFPSY1571) <-c("B6NFPSY1571")
B6NFPSY1571 <- as.data.frame(t(B6NFPSY1571))
LL1_3_S145 <- read.csv(file=file.choose())
rownames(LL1_3_S145) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                           "Segmentation_Average", "Segmentation_Median")
colnames(LL1_3_S145) <-c("LL1_3_S145")
LL1_3_S145 <- as.data.frame(t(LL1_3_S145))
LL1_4_C57BL6J <- read.csv(file=file.choose())
rownames(LL1_4_C57BL6J) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL1_4_C57BL6J) <-c("LL1_4_C57BL6J")
LL1_4_C57BL6J <- as.data.frame(t(LL1_4_C57BL6J))
LL2_2_S035 <- read.csv(file=file.choose())
rownames(LL2_2_S035) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                             "Segmentation_Average", "Segmentation_Median")
colnames(LL2_2_S035) <-c("LL2_2_S035")
LL2_2_S035 <- as.data.frame(t(LL2_2_S035))
LL2_2_B6N <- read.csv(file=file.choose())
rownames(LL2_2_B6N) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL2_2_B6N) <-c("LL2_2_B6N")
LL2_2_B6N <- as.data.frame(t(LL2_2_B6N))
LL2_3_S336 <- read.csv(file=file.choose())
rownames(LL2_3_S336) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                         "Segmentation_Average", "Segmentation_Median")
colnames(LL2_3_S336) <-c("LL2_3_S336")
LL2_3_S336 <- as.data.frame(t(LL2_3_S336))
LL2_4_BTBR <- read.csv(file=file.choose())
rownames(LL2_4_BTBR) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL2_4_BTBR) <-c("LL2_4_BTBR")
LL2_4_BTBR <- as.data.frame(t(LL2_4_BTBR))
LL2_4_NOR <- read.csv(file=file.choose())
rownames(LL2_4_NOR) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL2_4_NOR) <-c("LL2_4_NOR")
LL2_4_NOR <- as.data.frame(t(LL2_4_NOR))
LL4_1_S116 <- read.csv(file=file.choose())
rownames(LL4_1_S116) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                         "Segmentation_Average", "Segmentation_Median")
colnames(LL4_1_S116) <-c("LL4_1_S116")
LL4_1_S116 <- as.data.frame(t(LL4_1_S116))
LL4_1_S114 <- read.csv(file=file.choose())
rownames(LL4_1_S114) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL4_1_S114) <-c("LL4_1_S114")
LL4_1_S114 <- as.data.frame(t(LL4_1_S114))
LL4_1_AKR <- read.csv(file=file.choose())
rownames(LL4_1_AKR) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL4_1_AKR) <-c("LL4_1_AKR")
LL4_1_AKR <- as.data.frame(t(LL4_1_AKR))
LL5_4_S011 <- read.csv(file=file.choose())
rownames(LL5_4_S011) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                         "Segmentation_Average", "Segmentation_Median")
colnames(LL5_4_S011) <-c("LL5_4_S011")
LL5_4_S011 <- as.data.frame(t(LL5_4_S011))
LL6_2_S243 <- read.csv(file=file.choose())
rownames(LL6_2_S243) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(LL6_2_S243) <-c("LL6_2_S243")
LL6_2_S243 <- as.data.frame(t(LL6_2_S243))
WT37PSY <- read.csv(file=file.choose())
rownames(WT37PSY) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                          "Segmentation_Average", "Segmentation_Median")
colnames(WT37PSY) <-c("WT37PSY")
WT37PSY <- as.data.frame(t(WT37PSY))
WT28PSY <- read.csv(file=file.choose())
rownames(WT28PSY) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                       "Segmentation_Average", "Segmentation_Median")
colnames(WT28PSY) <-c("WT28PSY")
WT28PSY <- as.data.frame(t(WT28PSY))
WT27PSY <- read.csv(file=file.choose())
rownames(WT27PSY) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                       "Segmentation_Average", "Segmentation_Median")
colnames(WT27PSY) <-c("WT27PSY")
WT27PSY <- as.data.frame(t(WT27PSY))
WT1PSY <- read.csv(file=file.choose())
rownames(WT1PSY) <- c("Ellfit_Average", "Ellfit_Median", "Ellfit_Stretched", "Ellfit_Scrunched", "Ellfit_Average_Median",
                       "Segmentation_Average", "Segmentation_Median")
colnames(WT1PSY) <-c("WT1PSY")
WT1PSY <- as.data.frame(t(WT1PSY))
#turns out, only 22 videos produced a csv file so just going to work with those 22
sample22 <- rbind(B6J3, B6JFPSY, B6JMPSY, B6NFPSY1500, B6NFPSY1571,
                  LL1_3_S145, LL1_4_C57BL6J, LL2_2_B6N, LL2_2_S035, LL2_3_S336,
                  LL2_4_BTBR, LL2_4_NOR, LL4_1_AKR, LL4_1_S114, LL4_1_S116,
                  LL5_4_S011, LL6_2_S243, WT1PSY, WT27PSY, WT28PSY, WT37PSY)
sample22$weight <- c(30.1, 23.1, 25.6, 19.3, 21.5, 
                     23.1, 24.5, 21.8, 26.7, 27.6, 
                     32, 20, 25.6, 31.4, 20.7,
                     30.2, 35.6, 31.2, 24.3, 21.4, 31.2)
sample22$sex <- as.factor(c('M', 'F', 'M', 'F', 'F', 
                            'F', 'M', 'F', 'F', 'F',
                            'M', 'F', 'F', 'M', 'F',
                            'M', 'M', 'M', 'F', 'F', 'M'))

#plotting distributions 
hist(sample22$Ellfit_Average) #approximately normal
hist(sample22$Ellfit_Median) #approximately normal
hist(sample22$Segmentation_Average) #approximately normal but not as normal as ellfit
hist(sample22$Segmentation_Median) #approximately normal but not as normal as ellfit

#initial modeling for ellipse-fit and segmentation (average)
ellfitlm <- lm(sample22$weight~sample22$Ellfit_Average)
plot(sample22$weight~sample22$Ellfit_Average, 
     main="Initial Model for Ellipse-Fit",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=1,
     xlim=c(1000, 1800),
     ylim=c(15, 36))
abline(ellfitlm)
summary(ellfitlm)
sum_ellfitlm <- summary(ellfitlm)
mean(sum_ellfitlm$residuals^2)

seglm <- lm(sample22$weight~sample22$Segmentation_Average)
plot(sample22$weight~sample22$Segmentation_Average,
     main="Initial Model for Segmentation ",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=16,
     xlim=c(1000, 1800),
     ylim=c(15, 36))
abline(seglm)
summary(seglm)
sum_seglm <- summary(seglm)
mean(sum_seglm$residuals^2)

#initial modeling for ellipse-fit and segmentation (median)
ellfitlm_m <- lm(sample22$weight~sample22$Ellfit_Median)
plot(sample22$weight~sample22$Ellfit_Median, 
     main="Initial Model for Ellipse-Fit (Median)",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=1,
     xlim=c(1000, 1800),
     ylim=c(15, 36))
abline(ellfitlm_m)
summary(ellfitlm_m)

seglm_m <- lm(sample22$weight~sample22$Segmentation_Median)
plot(sample22$weight~sample22$Segmentation_Median,
     main="Initial Model for Segmentation (Median)",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=16,
     xlim=c(1000, 1800),
     ylim=c(15, 36))
abline(seglm_m)
summary(seglm_m)

#adding sex to the model (average)
ellfitlm_sex <- lm(sample22$weight~sample22$Ellfit_Average + sample22$sex)
plot(sample22$weight~sample22$Ellfit_Average, 
     main="Model for Ellipse-Fit w/ Sex",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=1,
     xlim=c(1000, 1800),
     ylim=c(15, 36),
     col=sample22$sex)
abline(ellfitlm_sex)
summary(ellfitlm_sex)
sum_ellfitlm_sex <- summary(ellfitlm_sex)
mean(sum_ellfitlm_sex$residuals^2)

seglm_sex <- lm(sample22$weight~sample22$Segmentation_Average + sample22$sex)
plot(sample22$weight~sample22$Segmentation_Average, 
     main="Model for Segmentation w/ Sex",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=16,
     xlim=c(1000, 1800),
     ylim=c(15, 36),
     col=sample22$sex)
abline(seglm_sex)
summary(seglm_sex)
sum_seglm_sex <- summary(seglm_sex)
mean(sum_seglm_sex$residuals^2)

##adding sex to the model (median)
ellfitlm_m_sex <- lm(sample22$weight~sample22$Ellfit_Median + sample22$sex)
plot(sample22$weight~sample22$Ellfit_Median, 
     main="Model for Ellipse-Fit w/Sex (Median)",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=1,
     xlim=c(1000, 1800),
     ylim=c(15, 36),
     col=sample22$sex)
abline(ellfitlm_m_sex)
summary(ellfitlm_m_sex)

seglm_m_sex <- lm(sample22$weight~sample22$Segmentation_Median + sample22$sex)
plot(sample22$weight~sample22$Segmentation_Median,
     main="Model for Segmentation w/ Sex (Median)",
     xlab="Average Pixel Count",
     ylab="Mouse Weight (g)",
     pch=16,
     xlim=c(1000, 1800),
     ylim=c(15, 36),
     col=sample22$sex)
abline(seglm_m_sex)
summary(seglm_m_sex)

#cross-validation using training and testing sets
train <- sample(22, 17)
train_ellfit <- lm(sample22$weight~sample22$Ellfit_Average + sample22$sex, 
                   subset=train)
summary(train_ellfit)
predict(train_ellfit, sample22)[-train]
mean((sample22$weight-predict(train_ellfit, sample22))[-train]^2)

train_seg <- lm(sample22$weight~sample22$Segmentation_Average + sample22$sex, 
                   subset=train)
summary(train_seg)
predict(train_seg, sample22)[-train]
mean((sample22$weight-predict(train_seg, sample22))[-train]^2)

train_m_ellfit <- lm(sample22$weight~sample22$Ellfit_Median + sample22$sex, 
                   subset=train)
summary(train_m_ellfit)
predict(train_m_ellfit, sample22)[-train]
mean((sample22$weight-predict(train_m_ellfit, sample22))[-train]^2)

train_m_seg <- lm(sample22$weight~sample22$Segmentation_Median + sample22$sex, 
                subset=train)
summary(train_m_seg)
predict(train_m_seg, sample22)[-train]
mean((sample22$weight-predict(train_m_seg, sample22))[-train]^2)

