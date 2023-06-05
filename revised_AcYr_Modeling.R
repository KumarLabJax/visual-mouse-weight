library(tidyverse)
library(patchwork)
library(plyr) 
library(reshape2)
library(ggpubr)
library(ggrepel)
library(ggforce)
library(rstatix)
library(RColorBrewer)
library(plotly)
library(Metrics)
library(zoo)
library(cowplot)
library(lubridate)
library(reshape2)
library(data.table)
library(dplyr)
# library(egg)
library(glmnet)
library(AICcmodavg)
library(lmtest)
library(ggpmisc)




##################### Reading and cleaning strain survey data ##################### 

### survey_with_corners_55min file is cropped, survey_with_corners is full data
# data <- read_csv("survey_with_corners.csv")
data <- read_csv("data/survey_with_corners_55min.csv")

# Filter out infinite elongation and bad areas
data <- data %>% 
  filter(!is.infinite(elongation))

data <- data %>% 
    filter(area>300)

#Add age to data, remove NA ages
data$Age = data$TestDate - data$DOB
data <- filter(data, !is.na(Age))

############################################################### 


##################### Models on full data and statistic tests ##################### 

### Models including box
base_model <- lm(Weight~area + Box, data = data)
norm_model <- lm(Weight~`norm area` + Box, data = data)
ecc_model <- lm(Weight~norm_eccen_area + Box, data = data)
nongen_model <- lm(Weight~norm_eccen_area + Box + Sex + Age, data = data)
full_model <- lm(Weight~norm_eccen_area +Box + Sex * Strain + Age, data = data)

# Residual analysis
par(mfrow=c(2,2))
plot(full_model)
par(mfrow=c(1,1))

se <- function(x) sd(x)/sqrt(length(x))

### Model summaries
summary(full_model)
### Model ANOVAs
anova(base_model)

m1 <- lm(Weight~area, data = data)
m2 <- lm(Weight~area + Box, data = data)
m3 <- lm(Weight~area * Box, data = data)

lrtest(m2,m3)

summary(m3)

############################################################### 



##################### Test/train split with no cross-validation (fig 3a-b) ##################### 

set.seed(1)

#create ID column
data$id <- 1:nrow(data)

#use 70% of dataset as training set and 30% as test set 
train <- data %>% group_by(Strain) %>% sample_frac(0.70)
test  <- dplyr::anti_join(data, train, by = 'id')


#Corner training models
train1 <- lm(Weight~area, data = train)
train2 <- lm(Weight~`norm area`, data = train)
train3 <- lm(Weight~`norm area` + Box, data = train)
train4 <- lm(Weight~norm_eccen_area + Box, data = train)
train5 <- lm(Weight~norm_eccen_area + Box + Sex + Age, data = train)
train6 <- lm(Weight~norm_eccen_area + Box + Sex * Strain + Age, data = train)

#Residual checking
par(mfrow=c(2,2))
plot(train2)
par(mfrow=c(1,1))

#Making predictions
ypred_M1 <- as.numeric(predict(train1, test))
ypred_M6 <- as.numeric(predict(train6, test))

plot_rmse_M1 <- paste("RMSE =", toString(round(rmse(test$Weight, ypred_M1),2)), "g")
plot_rmse_M6 <- paste("RMSE =", toString(round(rmse(test$Weight, ypred_M6),2)), "g")
plot_mae_M1 <- paste("MAE =", toString(round(mae(test$Weight, ypred_M1),2)), "g")
plot_mae_M6 <- paste("MAE =", toString(round(mae(test$Weight, ypred_M6),2)), "g")
plot_mape_M1 <- paste("MAPE =", toString(round(100*mape(test$Weight,ypred_M1),2)), "%")
plot_mape_M6 <- paste("MAPE =", toString(round(100*mape(test$Weight,ypred_M6),2)), "%")


#Full plot
M1_train_plot <- test %>% 
  ggplot(mapping = aes(x = Weight, y = ypred_M1)) +
  geom_point(color = '#009ED0', alpha = 0.5, size=1.5)+
  geom_abline(intercept = 0, slope = 1, size = 1.5, color='#05396B')+
  stat_regline_equation(label.x=8, label.y = 46.5, size=3, aes(label = ..rr.label..)) + 
  annotate("text", x = 8, y = 44, hjust = 0, size=3, label = plot_mae_M1) +
  annotate("text", x = 8, y = 41.5, hjust = 0, size=3, label = plot_rmse_M1) +
  annotate("text", x = 8, y = 39, hjust = 0, size=3, label = plot_mape_M1) +
  labs(x='True Mass (g)', y='Predicted Mass (g)') +
  coord_fixed() +
  ylim(7,48) +
  xlim(7,48) +
  theme_bw(base_size = 12)
  
M6_train_plot <- test %>% 
  ggplot(mapping = aes(x = Weight, y = ypred_M6)) +
  geom_point(color = '#009ED0', alpha = 0.5, size=1.5)+
  geom_abline(intercept = 0, slope = 1, size = 1.5, color='#05396B')+
  stat_regline_equation(label.x=8, label.y = 46.5, size=3, aes(label = ..rr.label..)) + 
  annotate("text", x = 8, y = 44, hjust = 0, size=3, label = plot_mae_M6) +
  annotate("text", x = 8, y = 41.5, hjust = 0, size=3, label = plot_rmse_M6) +
  annotate("text", x = 8, y = 39, hjust = 0, size=3, label = plot_mape_M6) +
  labs(x='True Mass (g)', y='Predicted Mass (g)') +
  coord_fixed() +
  ylim(7,48) +
  xlim(7,48) +
  theme_bw(base_size = 12)


ggarrange(M1_train_plot,M6_train_plot)


########################  Sex split fig (4a)
testF <- filter(test, Sex == "F")
testM <- filter(test, Sex == "M")

## Report on full model (M6)
ypred_allSex <- as.numeric(predict(train6, test))
ypredF <- as.numeric(predict(train6, testF))
ypredM <- as.numeric(predict(train6, testM))

plot_rmse_F <- paste("RMSE =", toString(round(rmse(testF$Weight, ypredF),3)), "g")
plot_rmse_M <- paste("RMSE =", toString(round(rmse(testM$Weight, ypredM),3)), "g")
plot_mae_F <- paste("MAE =", toString(round(mae(testF$Weight, ypredF),3)), "g")
plot_mae_M <- paste("MAE =", toString(round(mae(testM$Weight, ypredM),3)), "g")
plot_mape_F <- paste("MAPE =", toString(round(100*mape(testF$Weight,ypredF),2)), "%")
plot_mape_M <- paste("MAPE =", toString(round(100*mape(testM$Weight,ypredM),2)), "%")


r2_F <- round(cor(testF$Weight, ypredF)^2,3)
r2_M <- round(cor(testM$Weight, ypredM)^2,3)


r2_F_label = paste("italic(R) ^ 2 ==", r2_F) #sprintf("R^2 == %0.3f", r2_F)
r2_M_label = paste("italic(R) ^ 2 ==", r2_M) #sprintf("R^2 == %0.3f", r2_M)
sexpalette <- c("#e41a1c", "#377eb8")


full_label_F <- (paste("Female\n",r2_F_label,'\n',plot_mae_F,'\n',plot_rmse_F,'\n',plot_mape_F))

sex_plot <- test %>% 
  ggplot(mapping = aes(x = Weight, y = ypred_allSex, color = Sex)) +
  geom_point(alpha = 0.3, size=1.5)+
  geom_abline(intercept = 0, slope = 1, size = 1, color='black')+#05396B
  # stat_regline_equation(label.x=8, label.y = 46.5, size=5, aes(label = ..rr.label..)) +
  # stat_cor(size = 5, aes(label = paste(..rr.label.., sep = "*`,`~")),label.x.npc = 0.1)+
  # geom_label(aes(x = 7, y = 44, hjust = 0, size=0.25, label = full_label_F), parse=TRUE, color = sexpalette[1])+
  annotate("text", x = 7, y = 49, hjust = 0, size=3, label = "Female", color = sexpalette[1]) +
  annotate("text", x = 7, y = 46, hjust = 0, size=3, label = r2_F_label, parse=TRUE, color = sexpalette[1]) +
  annotate("text", x = 7, y = 43, hjust = 0, size=3, label = plot_mae_F, color = sexpalette[1]) +
  annotate("text", x = 7, y = 40, hjust = 0, size=3, label = plot_rmse_F, color = sexpalette[1]) +
  annotate("text", x = 7, y = 37, hjust = 0, size=3, label = plot_mape_F, color = sexpalette[1]) +
  annotate("text", x = 33, y = 19, hjust = 0, size=3, label = "Male", color = sexpalette[2]) +
  annotate("text", x = 33, y = 16, hjust = 0, size=3, label = r2_M_label, parse=TRUE, color = sexpalette[2]) +
  annotate("text", x = 33, y = 13, hjust = 0, size=3, label = plot_mae_M, color = sexpalette[2]) +
  annotate("text", x = 33, y = 10, hjust = 0, size=3, label = plot_rmse_M, color = sexpalette[2]) +
  annotate("text", x = 33, y = 7, hjust = 0, size=3, label = plot_mape_M, color = sexpalette[2]) +
  scale_color_manual(values=sexpalette)+
  labs(x='True Mass (g)', y='Predicted Mass (g)') +
  coord_fixed() +
  ylim(7,50) +
  xlim(7,50) +
  theme_bw(base_size = 12)+
  theme(panel.grid.minor=element_blank(),
        legend.position = 'none')

sex_plot




#faceted sex split plot, faceting drops the M rmse,mae
# train_plotF <- testF %>% 
#   ggplot(mapping = aes(x = Weight, y = ypredF)) +
#   geom_point(color = '#009ED0', alpha = 0.6, size=2.2)+
#   geom_abline(intercept = 0, slope = 1, size = 1.5, color='#05396B')+
#   stat_regline_equation(label.x=11, label.y = 45, size = 5, aes(label = ..rr.label..)) +
#   annotate("text", x = 11, y = 43, hjust = 0, size = 5, label = plot_mae_F) +
#   annotate("text", x = 11, y = 41, hjust = 0, size = 5, label = plot_rmse_F) +
#   annotate("text", x = 11, y = 39, hjust = 0, size = 5, label = plot_mape_F) +
#   labs(x='True Mass (g)', y='Predicted Mass (g)') +
#   ggtitle("Females") +
#   coord_fixed()+
#   ylim(7,48)+
#   xlim(7,48)+
#   theme_bw(base_size = 16)
# 
# 
# train_plotM <- testM %>% 
#   ggplot(mapping = aes(x = Weight, y = ypredM)) +
#   geom_point(color = '#009ED0', alpha = 0.6, size=2.2)+
#   geom_abline(intercept = 0, slope = 1, size = 1.5, color='#05396B')+
#   stat_regline_equation(label.x=11, label.y = 45, size = 5, aes(label = ..rr.label..)) + 
#   annotate("text", x = 11, y = 43, hjust = 0, size = 5, label = plot_mae_M) +
#   annotate("text", x = 11, y = 41, hjust = 0, size = 5, label = plot_rmse_M) +
#   annotate("text", x = 11, y = 39, hjust = 0, size = 5, label = plot_mape_M) +
#   labs(x='True Mass (g)', y='') +
#   ggtitle("Males") +
#   coord_fixed()+
#   ylim(7,48)+
#   xlim(7,48)+
#   theme_bw(base_size = 16)

# ggarrange(train_plotF,train_plotM, nrow = 1, ncol = 2)

######################################################################### 



########################## Strain-wise Average (Fig 4b) ########################## 

test$prediction <- as.numeric(predict(train6, test))
test$Weight

trimmed_obs_preds <- filter(select(test,c(Strain, Weight, prediction)), Strain != "NU/J")

Obs_mean <- tapply(trimmed_obs_preds$Weight, trimmed_obs_preds$Strain, mean)
Obs_sd <- tapply(trimmed_obs_preds$Weight, trimmed_obs_preds$Strain, sd)
Pred_mean <- tapply(trimmed_obs_preds$prediction, trimmed_obs_preds$Strain, mean)
Pred_sd <- tapply(trimmed_obs_preds$prediction, trimmed_obs_preds$Strain, sd)

df_Observed <- data.frame(Strain = names(Obs_mean), Obs_mean = Obs_mean, Obs_sd = Obs_sd)
df_Predicted <- data.frame(Strain = names(Pred_mean), Pred_mean = Pred_mean, Pred_sd = Pred_sd)

df_Observed
df_Predicted

mean_var_df <- merge(df_Observed, df_Predicted, by="Strain")

unique(is.na(Obs_sd))

labeled_strains <- c("C57BL/6J","C57BL/6NJ","BALB/cJ","A/J")
extreme_strains <- c("MSM/MsJ","NZO/HILtJ")


export_strain_errors <- mean_var_df
colnames(export_strain_errors) <- c("obs_mass_mean","obs_mass_sd","pred_mass_mean","pred_mass_sd")
write.csv(export_strain_errors, "strainwise_means_SDs_obs_preds.csv")
          
### Strainwise average 
strainwise_plot <- mean_var_df %>% 
  ggplot(mapping = aes(x = Obs_mean, y = Pred_mean, color = Strain, label=Strain)) +
  geom_abline(intercept = 0, slope = 1, size = 1, color='black')+
  geom_point(size=1.2)+
  # geom_text(hjust=0, vjust=0)+
  geom_errorbar(aes(ymin=Pred_mean-Pred_sd, ymax=Pred_mean+Pred_sd), width=1, size = 0.3) +
  geom_errorbarh(aes(xmin=Obs_mean-Obs_sd, xmax=Obs_mean+Obs_sd), height=1, size = 0.3) +
  labs(x='True Mass (g)', y='Predicted Mass (g)')+
  coord_fixed()+
  # coord_cartesian(xlim=c(7,48), ylim=c(7,48))+
  ylim(5,50)+
  xlim(5,50)+
  theme_bw(base_size = 12)+
  guides(colour = guide_legend(ncol = 5))+
  geom_label_repel(aes(label=ifelse((Strain %in% labeled_strains),as.character(Strain),''), hjust='center', vjust='center'),
                  box.padding   = 0.5,
                  point.padding = 0.5,
                  force = 1,
                  nudge_y       = 18,
                  nudge_x = -12,
                  direction     = "y",
                  segment.size  = 0.5,
                  size = 3)+
  geom_label_repel(aes(label=ifelse((Strain == "NZO/HILtJ"),as.character(Strain),''), hjust='center', vjust='center'),
                   box.padding   = 0.5,
                   point.padding = 0.5,
                   nudge_y       = -18,
                   nudge_x       = 6,
                   segment.size  = 0.5,
                   size = 3)+
  geom_label_repel(aes(label=ifelse((Strain == "MSM/MsJ"),as.character(Strain),''), hjust='center', vjust='center'),
                   box.padding   = 0.5,
                   # point.padding = 0.5,
                   nudge_y       = -2,
                   nudge_x       = 15,
                   segment.size  = 0.5,
                   size = 3)+
  theme(legend.position="none",
  panel.grid.minor=element_blank())


strainwise_plot


ggarrange(sex_plot, strainwise_plot, ncol = 2, nrow=1)



# train1_plot <- mean_var_df %>% 
#   ggplot(mapping = aes(x = Obs_mean, y = Pred_mean, color = Strain)) +
#   geom_abline(intercept = 0, slope = 1, size = 1, color='black')+
#   geom_point(size=2)+
#   geom_errorbar(aes(ymin=Pred_mean-Pred_sd, ymax=Pred_mean+Pred_sd), width=.5) +
#   geom_errorbarh(aes(xmin=Obs_mean-Obs_sd, xmax=Obs_mean+Obs_sd), height=.5) +
#   labs(x='True Weight (g)', y='Predicted Weight (g)')+
#   ggtitle("Model 1 Test")+
#   theme_bw()+
#   annotate("text", x = 48, y = 50, angle = 45, label = '45 deg', size = 5)+
#   stat_regline_equation(label.x=10, label.y = 34, aes(label = ..adj.rr.label..)) +
#   stat_regline_equation(label.x=10, label.y = 37, aes(label = ..eq.label..)) 
#   

############################################################### 


######################## 50-fold cross-validation (Fig 3c-d) ########################  
niter <- 50

RMSE1 <- rep(0,niter)
RMSE2 <- rep(0,niter)
RMSE3 <- rep(0,niter)
RMSE4 <- rep(0,niter)
RMSE5 <- rep(0,niter)
RMSE6 <- rep(0,niter)

MAE1 <- rep(0,niter)
MAE2 <- rep(0,niter)
MAE3 <- rep(0,niter)
MAE4 <- rep(0,niter)
MAE5 <- rep(0,niter)
MAE6 <- rep(0,niter)

MAPE1 <- rep(0,niter)
MAPE2 <- rep(0,niter)
MAPE3 <- rep(0,niter)
MAPE4 <- rep(0,niter)
MAPE5 <- rep(0,niter)
MAPE6 <- rep(0,niter)

R2_1 <- rep(0,niter)
R2_2 <- rep(0,niter)
R2_3 <- rep(0,niter)
R2_4 <- rep(0,niter)
R2_5 <- rep(0,niter)
R2_6 <- rep(0,niter)


#n-fold cross 
for (iter in 1:niter){
  set.seed(iter)
  
  #create ID column
  data$id <- 1:nrow(data)
  
  #use 70% of dataset as training set and 30% as test set 
  train_data <- data %>% group_by(Strain) %>% sample_frac(0.70)
  test_data  <- dplyr::anti_join(data, train_data, by = join_by('id'))
  
  #Training models, comment out ones you don't want to run
  
  train1 <- lm(Weight~area, data = train_data)
  train2 <- lm(Weight~`norm area`, data = train_data)
  train3 <- lm(Weight~`norm area` + Box, data = train_data)
  train4 <- lm(Weight~norm_eccen_area + Box, data = train_data)
  train5 <- lm(Weight~norm_eccen_area + Box + Sex + Age, data = train_data)
  train6 <- lm(Weight~norm_eccen_area +Box + Sex * Strain + Age, data = train_data)
  
  #predictions
  ypred1 <- as.numeric(predict(train1, test_data))
  ypred2 <- as.numeric(predict(train2, test_data))
  ypred3 <- as.numeric(predict(train3, test_data))
  ypred4 <- as.numeric(predict(train4, test_data))
  ypred5 <- as.numeric(predict(train5, test_data))
  ypred6 <- as.numeric(predict(train6, test_data))

  RMSE1[iter] <- round(rmse(test_data$Weight, ypred1),3)
  RMSE2[iter] <- round(rmse(test_data$Weight, ypred2),3)
  RMSE3[iter] <- round(rmse(test_data$Weight, ypred3),3)
  RMSE4[iter] <- round(rmse(test_data$Weight, ypred4),3)
  RMSE5[iter] <- round(rmse(test_data$Weight, ypred5),3)
  RMSE6[iter] <- round(rmse(test_data$Weight, ypred6),3)

  MAE1[iter] <- round(mae(test_data$Weight, ypred1),3)
  MAE2[iter] <- round(mae(test_data$Weight, ypred2),3)
  MAE3[iter] <- round(mae(test_data$Weight, ypred3),3)
  MAE4[iter] <- round(mae(test_data$Weight, ypred4),3)
  MAE5[iter] <- round(mae(test_data$Weight, ypred5),3)
  MAE6[iter] <- round(mae(test_data$Weight, ypred6),3)
  
  MAPE1[iter] <- round(100*mape(test_data$Weight,ypred1),3)
  MAPE2[iter] <- round(100*mape(test_data$Weight,ypred2),3)
  MAPE3[iter] <- round(100*mape(test_data$Weight,ypred3),3)
  MAPE4[iter] <- round(100*mape(test_data$Weight,ypred4),3)
  MAPE5[iter] <- round(100*mape(test_data$Weight,ypred5),3)
  MAPE6[iter] <- round(100*mape(test_data$Weight,ypred6),3)

  R2_1[iter] <- round(cor(test_data$Weight, ypred1)^2,3)
  R2_2[iter] <- round(cor(test_data$Weight, ypred2)^2,3)
  R2_3[iter] <- round(cor(test_data$Weight, ypred3)^2,3)
  R2_4[iter] <- round(cor(test_data$Weight, ypred4)^2,3)
  R2_5[iter] <- round(cor(test_data$Weight, ypred5)^2,3)
  R2_6[iter] <- round(cor(test_data$Weight, ypred6)^2,3)
}


all_RMSE <- c(mean(RMSE1), mean(RMSE2), mean(RMSE3), mean(RMSE4), mean(RMSE5), mean(RMSE6))
all_MAE <- c(mean(MAE1), mean(MAE2), mean(MAE3), mean(MAE4), mean(MAE5), mean(MAE6))
all_MAPE <- c(mean(MAPE1), mean(MAPE2), mean(MAPE3), mean(MAPE4), mean(MAPE5), mean(MAPE6))
all_R2 <- c(mean(R2_1), mean(R2_2), mean(R2_3), mean(R2_4), mean(R2_5), mean (R2_6))

RMSE_sd <- round(c(sd(RMSE1), sd(RMSE2), sd(RMSE3), sd(RMSE4), sd(RMSE5), sd(RMSE6)),3)
MAE_sd <- round(c(sd(MAE1), sd(MAE2), sd(MAE3), sd(MAE4), sd(MAE5), sd(MAE6)),3)
MAPE_sd <- round(c(sd(MAPE1), sd(MAPE2), sd(MAPE3), sd(MAPE4), sd(MAPE5), sd(MAPE6)),3)
R2_sd <- round(c(sd(R2_1), sd(R2_2), sd(R2_3), sd(R2_4), sd(R2_5), sd(R2_6)),3)


### Prints table 2 one row at a time
str_c(all_RMSE[6]," ± ",RMSE_sd[6]," & ",all_MAE[6]," ± ",MAE_sd[6]," & ",
      all_MAPE[6]," ± ",MAPE_sd[6]," & ",all_R2[6]," ± ",R2_sd[6])


########################## End of regular cross-validation ########################## 


########################## Separate cross-validation for table 1 ########################## 

niter <- 50

tb1_RMSE1 <- rep(0,niter)
tb1_RMSE2 <- rep(0,niter)
tb1_RMSE3 <- rep(0,niter)
tb1_RMSE4 <- rep(0,niter)
tb1_RMSE5 <- rep(0,niter)
tb1_RMSE6 <- rep(0,niter)

tb1_MAE1 <- rep(0,niter)
tb1_MAE2 <- rep(0,niter)
tb1_MAE3 <- rep(0,niter)
tb1_MAE4 <- rep(0,niter)
tb1_MAE5 <- rep(0,niter)
tb1_MAE6 <- rep(0,niter)

tb1_MAPE1 <- rep(0,niter)
tb1_MAPE2 <- rep(0,niter)
tb1_MAPE3 <- rep(0,niter)
tb1_MAPE4 <- rep(0,niter)
tb1_MAPE5 <- rep(0,niter)
tb1_MAPE6 <- rep(0,niter)

tb1_R2_1 <- rep(0,niter)
tb1_R2_2 <- rep(0,niter)
tb1_R2_3 <- rep(0,niter)
tb1_R2_4 <- rep(0,niter)
tb1_R2_5 <- rep(0,niter)
tb1_R2_6 <- rep(0,niter)


#n-fold cross 
for (iter in 1:niter){
  set.seed(iter)
  
  #create ID column
  data$id <- 1:nrow(data)
  
  #use 70% of dataset as training set and 30% as test set 
  train_data <- data %>% group_by(Strain) %>% sample_frac(0.70)
  test_data  <- dplyr::anti_join(data, train_data, by = join_by('id'))
  

  #Table 1 metric comparison
  eccen <- lm(Weight~norm_eccen_area, data = train)
  aspect <- lm(Weight~norm_aspect_area, data = train)
  elong <- lm(Weight~norm_elong_area, data = train)
  eccen_box <- lm(Weight~norm_eccen_area + Box, data = train)
  aspect_box <- lm(Weight~norm_aspect_area + Box, data = train)
  elong_box <- lm(Weight~norm_elong_area + Box, data = train)

  #Table 1 predictions
  ypred_eccen <- as.numeric(predict(eccen, test_data))
  ypred_aspect <- as.numeric(predict(aspect, test_data))
  ypred_elong <- as.numeric(predict(elong, test_data))
  ypred_eccen_box <- as.numeric(predict(eccen_box, test_data))
  ypred_aspect_box <- as.numeric(predict(aspect_box, test_data))
  ypred_elong_box <- as.numeric(predict(elong_box, test_data))
  
  tb1_RMSE1[iter] <- round(rmse(test_data$Weight, ypred_eccen),3)
  tb1_RMSE2[iter] <- round(rmse(test_data$Weight, ypred_aspect),3)
  tb1_RMSE3[iter] <- round(rmse(test_data$Weight, ypred_elong),3)
  tb1_RMSE4[iter] <- round(rmse(test_data$Weight, ypred_eccen_box),3)
  tb1_RMSE5[iter] <- round(rmse(test_data$Weight, ypred_aspect_box),3)
  tb1_RMSE6[iter] <- round(rmse(test_data$Weight, ypred_elong_box),3)
  
  tb1_MAE1[iter] <- round(mae(test_data$Weight, ypred_eccen),3)
  tb1_MAE2[iter] <- round(mae(test_data$Weight, ypred_aspect),3)
  tb1_MAE3[iter] <- round(mae(test_data$Weight, ypred_elong),3)
  tb1_MAE4[iter] <- round(mae(test_data$Weight, ypred_eccen_box),3)
  tb1_MAE5[iter] <- round(mae(test_data$Weight, ypred_aspect_box),3)
  tb1_MAE6[iter] <- round(mae(test_data$Weight, ypred_elong_box),3)
  
  tb1_MAPE1[iter] <- round(100*mape(test_data$Weight,ypred_eccen),3)
  tb1_MAPE2[iter] <- round(100*mape(test_data$Weight,ypred_aspect),3)
  tb1_MAPE3[iter] <- round(100*mape(test_data$Weight,ypred_elong),3)
  tb1_MAPE4[iter] <- round(100*mape(test_data$Weight,ypred_eccen_box),3)
  tb1_MAPE5[iter] <- round(100*mape(test_data$Weight,ypred_aspect_box),3)
  tb1_MAPE6[iter] <- round(100*mape(test_data$Weight,ypred_elong_box),3)
  
  tb1_R2_1[iter] <- round(cor(test_data$Weight, ypred_eccen)^2,3)
  tb1_R2_2[iter] <- round(cor(test_data$Weight, ypred_aspect)^2,3)
  tb1_R2_3[iter] <- round(cor(test_data$Weight, ypred_elong)^2,3)
  tb1_R2_4[iter] <- round(cor(test_data$Weight, ypred_eccen_box)^2,3)
  tb1_R2_5[iter] <- round(cor(test_data$Weight, ypred_aspect_box)^2,3)
  tb1_R2_6[iter] <- round(cor(test_data$Weight, ypred_elong_box)^2,3)
}


tb1_all_RMSE <- c(mean(tb1_RMSE1), mean(tb1_RMSE2), mean(tb1_RMSE3), mean(tb1_RMSE4), mean(tb1_RMSE5), mean(tb1_RMSE6))
tb1_all_MAE <- c(mean(tb1_MAE1), mean(tb1_MAE2), mean(tb1_MAE3), mean(tb1_MAE4), mean(tb1_MAE5), mean(tb1_MAE6))
tb1_all_MAPE <- c(mean(tb1_MAPE1), mean(tb1_MAPE2), mean(tb1_MAPE3), mean(tb1_MAPE4), mean(tb1_MAPE5), mean(tb1_MAPE6))
tb1_all_R2 <- c(mean(tb1_R2_1), mean(tb1_R2_2), mean(tb1_R2_3),mean(tb1_R2_4), mean(tb1_R2_5), mean(tb1_R2_6))

tb1_RMSE_sd <- round(c(sd(tb1_RMSE1), sd(tb1_RMSE2), sd(tb1_RMSE3), sd(tb1_RMSE4), sd(tb1_RMSE5), sd(tb1_RMSE6)),3)
tb1_MAE_sd <- round(c(sd(tb1_MAE1), sd(tb1_MAE2), sd(tb1_MAE3), sd(tb1_MAE4), sd(tb1_MAE5), sd(tb1_MAE6)),3)
tb1_MAPE_sd <- round(c(sd(tb1_MAPE1), sd(tb1_MAPE2), sd(tb1_MAPE3), sd(tb1_MAPE4), sd(tb1_MAPE5), sd(tb1_MAPE6)),3)
tb1_R2_sd <- round(c(sd(tb1_R2_1), sd(tb1_R2_2), sd(tb1_R2_3), sd(tb1_R2_4), sd(tb1_R2_5), sd(tb1_R2_6)),3)


k=6
str_c(tb1_all_RMSE[k]," ± ",tb1_RMSE_sd[k]," & ",tb1_all_MAE[k]," ± ",tb1_MAE_sd[k]," & ",
      tb1_all_MAPE[k]," ± ",tb1_MAPE_sd[k]," & ",tb1_all_R2[k]," ± ",tb1_R2_sd[k])

##################### ##################### ##################### ##################### 


########################## Box-plots for model comparison (fig 3c-d) ########################## 

#Requires cross-validation
MAE_df <- data.frame(MAE1,MAE2,MAE3,MAE4,MAE5,MAE6)
R2_df <- data.frame(R2_1,R2_2,R2_3,R2_4,R2_5,R2_6)

clean_MAE_df <- pivot_longer(MAE_df, cols=1:6, names_to = "model", values_to = "mae")
clean_R2_df <- pivot_longer(R2_df, cols=1:6, names_to = "model", values_to = "r2")

clean_MAE_df

old_labs <- c("Base", "+Unit", "+Box", "+Ecc", "+Sex\n+Age", "+Strain")
new_labs <- c("Base\n(M1)", "M2", "M3", "M4", "M5", "Full\n(M6)")


MAE_boxplot <- clean_MAE_df %>% 
  ggplot(mapping = aes(x = model, y = mae)) +
  geom_boxplot(color='#05396B', outlier.shape = NA, size = 0.3)+
  geom_jitter(color = '#009ED0',width=0.1,alpha=0.4, size=1.25)+
  labs(x="", y='MAE (g)') +
  coord_fixed(5)+
  theme_bw(base_size = 12)+
  scale_x_discrete("", labels = new_labs)


R2_boxplot <- clean_R2_df %>% 
  ggplot(mapping = aes(x = model, y = r2)) +
  geom_boxplot(color='#05396B', outlier.shape = NA, size=0.3)+
  geom_jitter(color = '#009ED0',width=0.1,alpha=0.4, size=1.25)+
  labs(x="", y=expression(R^2 ~ Value)) +
  coord_fixed(24)+
  theme_bw(base_size=12)+
  scale_x_discrete("", labels = new_labs)


#Prints panels 3c-d together
ggarrange(M1_train_plot, M6_train_plot, MAE_boxplot, R2_boxplot, 
          ncol=2, nrow=2, labels=c("A","B","C","D"), align = "hv", 
          font.label=list(size = 18, color = "black", face="bold"))


# MAE_violin <- clean_MAE_df %>% 
#   ggplot(mapping = aes(x = model, y = mae)) +
#   geom_violin(fill='#05396B')+
#   # geom_jitter(color = '#009ED0',width=0.2,alpha=0.5)+
#   # geom_point(color = '#009ED0', alpha = 0.6, size=2.2)+
#   # geom_abline(intercept = 0, slope = 1, size = 1.5, color='#05396B')+
#   labs(x="", y='Mean Average Error (MAE) (g)') +
#   # ggtitle("Full Model") +
#   theme_bw()
# 
# MAE_violin
# 
# 
# R2_violin <- clean_R2_df %>% 
#   ggplot(mapping = aes(x = model, y = r2)) +
#   geom_violin(fill='#05396B')+
#   # geom_jitter(color = '#009ED0',width=0.2,alpha=0.5)+
#   # geom_point(color = '#009ED0', alpha = 0.6, size=2.2)+
#   # geom_abline(intercept = 0, slope = 1, size = 1.5, color='#05396B')+
#   labs(x="", y='R^2') +
#   # ggtitle("Full Model") +
#   theme_bw()
# 
# R2_violin


########################## End of Box-plots for model comparison ########################## 

  



############################  individual mouse area plots (fig 1) ############################ 
b6j_framedata <- filter(read_csv("data/1b_moments_data/LL1-1_C57BL6J.avi_moments.csv"), seg_area > 300)
b6n_framedata <- filter(read_csv("data/1b_moments_data/LL3-1_B6N_M.avi_moments.csv"), seg_area > 300)
aj_framedata <- filter(read_csv("data/1b_moments_data/LL1-4_AJ_M.avi_moments.csv"), seg_area > 300)
balb_framedata <- filter(read_csv("data/1b_moments_data/LL1-3_000651-M-MP16-9-42395-3-S009.avi_moments.csv"), seg_area > 300)


# # takes a df as x
# framestat <- function(x) { 
#   fd_mean <- mean(x$seg_area)
#   x$area_deviance <- x$seg_area - fd_mean
#   x$area_percentdev <- 100*x$area_deviance / fd_mean
#   return (x)
# }

framerate <- 30

fd_mean_1 <- mean(b6j_framedata$seg_area)
sd_1 <- sd(b6j_framedata$seg_area)
b6j_framedata$area_deviance <- b6j_framedata$seg_area - fd_mean_1
b6j_framedata$area_percentdev <- 100*b6j_framedata$area_deviance / fd_mean_1
b6j_framedata$minute <- b6j_framedata$frame / (framerate*60)


fd_mean_2 <- mean(b6n_framedata$seg_area)
sd_2 <- sd(b6n_framedata$seg_area)
b6n_framedata$area_deviance <- b6n_framedata$seg_area - fd_mean_2
b6n_framedata$area_percentdev <- 100*b6n_framedata$area_deviance / fd_mean_2
b6n_framedata$minute <- b6n_framedata$frame / (framerate*60)


fd_mean_3 <- mean(aj_framedata$seg_area)
sd_3 <- sd(aj_framedata$seg_area)
aj_framedata$area_deviance <- aj_framedata$seg_area - fd_mean_3
aj_framedata$area_percentdev <- 100*aj_framedata$area_deviance / fd_mean_3
aj_framedata$minute <- aj_framedata$frame / (framerate*60)


fd_mean_4 <- mean(balb_framedata$seg_area)
sd_4 <- sd(balb_framedata$seg_area)
balb_framedata$area_deviance <- balb_framedata$seg_area - fd_mean_4
balb_framedata$area_percentdev <- 100*balb_framedata$area_deviance / fd_mean_4
balb_framedata$minute <- balb_framedata$frame / (framerate*60)


b6j_frameplot <- b6j_framedata %>% 
  ggplot(mapping = aes(x = minute, y = area_percentdev)) +
  geom_line(aes(y=area_percentdev), color = '#009ED0', size=0.3)+
  # ggtitle("C57BL/6J")+
  ggtitle("C57BL/6J", subtitle= paste("Mass: 25.1g, Area:", round(fd_mean_1),"±",round(sd_1),"px"))+
  # labs(x='Time (min)', y='% Deviance')+ #between pixel area and mean pixel area')+
  labs(x='', y='% Deviance')+
  # annotate("text", x = 40, y = -60, label = paste("Mean area:\n", round(fd_mean_1),"px"), size = 6)+
  xlim(2,55)+
  ylim(-75,75)+
  geom_hline(yintercept = 0, size = 1, color='black')+
  # coord_cartesian(xlim=c(25000,75000), ylim=c(-50,50))+
  # coord_cartesian(xlim=c(0,60), ylim=c(-75,75))+
  theme_bw(base_size = 18)+
  theme(aspect.ratio=0.3)+
  theme(axis.ticks=element_line(size=1), 
        panel.grid.minor=element_blank(),
        plot.title = element_text(size=18),
        plot.subtitle = element_text(size=12))

b6n_frameplot <- b6n_framedata %>% 
  ggplot(mapping = aes(x = minute, y = area_percentdev)) +
  geom_line(aes(y=area_percentdev), color = '#009ED0', size=0.3)+
  # ggtitle("C57BL/6NJ")+
  ggtitle("C57BL/6NJ", subtitle= paste("Mass: 25.4g, Area:", round(fd_mean_2),"±",round(sd_2),"px"))+
  # labs(x='Time (min)', y='% Deviance')+ #between pixel area and mean pixel area')+
  labs(x='', y='% Deviance')+
  # annotate("text", x = 40, y = -60, label = paste("Mean area:\n", round(fd_mean_2),"px"), size = 6)+
  xlim(2,55)+
  ylim(-75,75)+
  geom_hline(yintercept = 0, size = 1, color='black')+
  # coord_cartesian(xlim=c(25000,75000), ylim=c(-50,50))+
  # coord_cartesian(xlim=c(0,60), ylim=c(-75,75))+
  theme_bw(base_size = 18)+
  theme(aspect.ratio=0.3)+
  theme(axis.ticks=element_line(size=1), 
        panel.grid.minor=element_blank(),
        plot.title = element_text(size=18),
        plot.subtitle = element_text(size=12))

aj_frameplot <- aj_framedata %>% 
  ggplot(mapping = aes(x = minute, y = area_percentdev)) +
  geom_line(aes(y=area_percentdev), color = '#009ED0', size=0.3)+
  # ggtitle("A/J")+
  ggtitle("A/J", subtitle= paste("Mass: 24.5g, Area:", round(fd_mean_3),"±",round(sd_3),"px"))+
  labs(x='', y='% Deviance')+ #between pixel area and mean pixel area')+
  # annotate("text", x = 40, y = -60, label = paste("Mean area:\n", round(fd_mean_3),"px"), size = 6)+
  xlim(2,55)+
  ylim(-75,75)+
  geom_hline(yintercept = 0, size = 1, color='black')+
  # coord_cartesian(xlim=c(25000,75000), ylim=c(-50,50))+
  # coord_cartesian(xlim=c(0,60), ylim=c(-75,75))+
  theme_bw(base_size = 18)+
  theme(aspect.ratio=0.3)+
  theme(axis.ticks=element_line(size=1), 
        panel.grid.minor=element_blank(),
        plot.title = element_text(size=18),
        plot.subtitle = element_text(size=12))


balb_frameplot <- balb_framedata %>% 
  ggplot(mapping = aes(x = minute, y = area_percentdev)) +
  geom_line(aes(y=area_percentdev), color = '#009ED0', size=0.3)+
  # ggtitle("BALB/cJ")+
  ggtitle("BALB/cJ", subtitle= paste("Mass: 25.5g, Area:", round(fd_mean_4),"±",round(sd_4),"px"))+
  # labs(x='Time (min)', y='% Deviance')+ #between pixel area and mean pixel area')+
  labs(x='Time (min)', y='% Deviance')+
  # annotate("text", x = 40, y = -60, label = paste("Mean area:\n", round(fd_mean_4),"px"), size = 6)+
  xlim(2,55)+
  ylim(-75,75)+
  geom_hline(yintercept = 0, size = 1, color='black')+
  # coord_cartesian(xlim=c(25000,75000), ylim=c(-50,50))+
  # coord_cartesian(xlim=c(0,60), ylim=c(-75,75))+
  theme_bw(base_size = 18)+
  theme(aspect.ratio=0.3)+
  theme(axis.ticks=element_line(size=1), 
        panel.grid.minor=element_blank(),
        plot.title = element_text(size=18),
        plot.subtitle = element_text(size=12),
        axis.title.x = element_text(size=16))


# Prints 4 plots of fig 1b together (not including pictures)
ggarrange(b6j_frameplot, b6n_frameplot, aj_frameplot, balb_frameplot, ncol = 1, nrow = 4)

###################################################################### 



############################ Relative sd boxplot (fig 1c) ############################ 
rsd_data <- read_csv("data/rsd_data/rsd_ecc_fullstrainsurvey.csv")[2:4]
colnames(rsd_data) <- c("NetworkFilename", "rsd", "ecc_rsd")
rsd_data$NetworkFilename <- gsub("_moments.csv", "", rsd_data$NetworkFilename)

straindata <- data
straindata$NetworkFilename <- gsub("^.*/","", straindata$NetworkFilename) #str_split(straindata$NetworkFilename, "/")

###straindata is the same as the full data, but with the path in network filename cut off
strainsurvey_withrsd  <- merge(x = straindata, y = rsd_data, by = 'NetworkFilename', all.x = TRUE)
strainsurvey_withrsd$Strain = with(strainsurvey_withrsd, reorder(Strain, rsd, median))


rsd_data

mean(rsd_data$rsd)
mean(rsd_data$ecc_rsd)

b6js <- filter(strainsurvey_withrsd, Strain == "C57BL/6J")
b6ns <- filter(strainsurvey_withrsd, Strain == "C57BL/6NJ")
pwks <- filter(strainsurvey_withrsd, Strain == "PWK/PhJ")


# rsdcolors <- c("Area" = "red", "Eccentric Area" = "blue")
# rsdcolors <- c("#ad2e24", "#1b065e")
# rsdcolors <- c("#e41a1c", "#377eb8")
rsdcolors <- c("#e41a1c", "#009ED0")
# d91e36 dd1c1a

rsd_boxplot <- strainsurvey_withrsd %>% 
  ggplot(aes(x = Strain)) +
  geom_boxplot(aes(y = ecc_rsd), fill = rsdcolors[1], size = 0.4,alpha=1, position = position_dodge(1))+
  geom_boxplot(aes(y = rsd), fill = rsdcolors[2], size = 0.4,alpha=0.8, position = position_dodge(1))+
  labs(x="", y='RSD')+
  theme_bw(base_size=16)+
  theme(legend.position = 'none')+
  ylim(0.05,0.21)+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=0.5, size=10))
# coord_flip()+
# theme(axis.text.x = element_text(size=14))+
# theme(axis.text.y = element_text(angle=0, hjust=1, vjust=0.5, size=9))

rsd_boxplot



# rsd_hist <- strainsurvey_withrsd %>% 
#   ggplot(mapping = aes(x = rsd, fill = Strain)) +
#   geom_histogram(binwidth = 0.001)+
#   labs(x="Relative SD of Area", y="Frequency") +
#   xlim(0,0.8)+
#   ylim(0,50)+
#   # scale_x_continuous(breaks=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8), limits=c(0,0.8))+
#   theme(panel.grid.minor = element_blank())+
#   theme_bw(base_size=16)+
#   coord_fixed(1/(60))
# # geom_vline(xintercept = median(rsd_data$rsd), size=1, color='red')




#################################### Misc stuff, Gautam & Brian code #################################### 

#Model 1
ypred <- as.numeric(predict(train_model1, test))
plot(ypred, test$Weight)

cor(ypred, test$Weight)^2

sum((ypred-test$Weight)^2)/(nrow(test)-1)

rmse(test$Weight, ypred)
mse(test$Weight, ypred)
mae(test$Weight, ypred)


mae(ypred, test$Weight)
?mae


sum(abs(ypred-test$Weight)/(nrow(test)-1))


plot(density((ypred-test$Weight)^2))

plot(density(abs(ypred-test$Weight)))

test_model1




#Model 2

#Predicts on test data
ypred <- as.numeric(predict(train_model2, test))
plot(ypred, test$Weight)


ypred_male <- as.numeric(predict(train_model2, filter(test, Sex=="M")))
ypred_female <- as.numeric(predict(train_model2, filter(test, Sex=="F")))


accuracyPlot <- test %>% 
  ggplot(mapping = aes(x = Weight, y = ypred)) +
  geom_point()+
  labs(x='True Weight (g)', y='Predicted Weight (g)')+
  ggtitle("Model 10 Test Data")+
  geom_abline(intercept = 0, slope = 1, size = 1, color='tomato')+
  geom_text(aes(42, 43, label = "45 deg", angle = 45), size=5)+
  geom_smooth(method=lm, se=TRUE)+
  stat_regline_equation(label.y = 36, aes(label = ..eq.label..)) +
  stat_regline_equation(label.y = 33, aes(label = ..rr.label..))

accuracyPlot


accuracyPlot_M <- filter(test, Sex=='M') %>% 
  ggplot(mapping = aes(x = Weight, y = ypred_male)) +
  geom_point()+
  labs(x='True Weight (g)', y='Predicted Weight (g)')+
  geom_abline(intercept = 0, slope = 1, size = 1, color='tomato')+
  geom_smooth(method=lm, se=TRUE)+
  stat_regline_equation(label.y = 36, aes(label = ..eq.label..)) +
  stat_regline_equation(label.y = 35, aes(label = ..rr.label..))

accuracyPlot_M


accuracyPlot_F <- filter(test, Sex=='F') %>% 
  ggplot(mapping = aes(x = Weight, y = ypred_female)) +
  geom_point()+
  labs(x='True Weight (g)', y='Predicted Weight (g)')+
  geom_abline(intercept = 0, slope = 1, size = 1, color='tomato')+
  # geom_smooth(method=lm, se=TRUE)+
  # stat_regline_equation(label.y = 36, aes(label = ..eq.label..)) +
  stat_regline_equation(label.y = 35, aes(label = ..rr.label..))

accuracyPlot_F



cor(ypred, test$Weight)^2



#Error tests:
rmse(test$Weight, ypred)
mse(test$Weight, ypred)
mae(test$Weight, ypred)
rsquare(train_model2, test)


#Error Graphs:
ggplot(data = test, aes(x=ypred)) +
  geom_density()

ggplot(data = test, aes(x=test$Weight)) +
  geom_density()

ggplot(data = test) +
  geom_density(aes(x=Weight, fill='True weight'), alpha=0.5)+
  geom_density(aes(x=ypred, fill='Predicted weight'), alpha=0.5)+
  ggtitle("Model 10 Test Data")


 ggplot(data = filter(test, Sex=="F")) +
  geom_density(aes(x=Weight, fill='True weight'), alpha=0.5)+
  geom_density(aes(x=ypred_female, fill='Predicted weight'), alpha=0.5)

ggplot(data = filter(test, Sex=="M")) +
  geom_density(aes(x=Weight, fill='True weight'), alpha=0.5)+
  geom_density(aes(x=ypred_male, fill='Predicted weight'), alpha=0.5)




plot(density(mse(train_model2, test)))


sum((ypred-test$Weight)^2)/(nrow(test)-1)

sum(abs(ypred-test$Weight)/(nrow(test)-1))

?density
?geom_density


plot(density((ypred-test$Weight)^2))

plot(density(abs(ypred-test$Weight)))




# Model 3


ypred_model3 <- as.numeric(predict(train_model3, test))
plot(ypred_model3, test$Weight)



plot3_rmse <- paste("RMSE =", toString(round(rmse(test$Weight, ypred_model3),3)))
plot3_mae <- paste("MAE =", toString(round(mae(test$Weight, ypred_model3),3)))


accuracyPlot_3 <- test %>% 
  ggplot(mapping = aes(x = Weight, y = ypred_model3)) +
  geom_point()+
  labs(x='True Weight (g)', y='Predicted Weight (g)')+
  ggtitle("Model 11 Test (Area*eccentricity, sex, strain, age, box)")+
  geom_abline(intercept = 0, slope = 1, size = 1, color='tomato')+
  geom_text(aes(50, 50, label = "45 deg", angle = 45), size=5)+
  #text(aes(0, 45, label = plot3_rmse), size=3)+
  # geom_text(aes(0, 42, label = plot3_mae), size=5)+
  #geom_smooth(method=lm, se=TRUE)+
  stat_regline_equation(label.x=10, label.y = 36, aes(label = ..eq.label..)) +
  stat_regline_equation(label.x=10, label.y = 34, aes(label = ..adj.rr.label..)) +
  annotate("text", x = 12, y = 39, label = plot3_rmse) +
  annotate("text", x = 12, y = 41, label = plot3_mae) 
  # stat_regline_equation(label.y = 39, aes(label = plot3_mae)) + 
  # stat_regline_equation(label.y = 40, aes(label = plot3_rmse))

accuracyPlot_3

rmse(test$Weight, ypred_model3)
mae(test$Weight, ypred_model3)






















#REDO


# Experiment 4.4 Models
simple_model <- lm(Weight~area, data = data)
visual_model1 <- lm(Weight~norm_area, data = data)
visual_model2 <- lm(Weight~norm_area + Box, data = data)
nongenetic_model <- lm(Weight~norm_area + Sex + Age + Box, data = data)
fullmodel_nobox <- lm(Weight~norm_area + Sex * Strain + Age, data = data)
full_model <- lm(Weight~norm_area + Sex * Strain + Box + Age, data = data)


# summary(full_model)

#Residual Plots
# par(mfrow=c(2,2))
# plot(full_model)
# par(mfrow=c(1,1))


## Training on 70% of data, rerunning model and testing on random 30%
set.seed(1)

#create ID column
data$id <- 1:nrow(data)

#use 70% of dataset as training set and 30% as test set
train <- data %>% group_by(Strain) %>% sample_frac(0.70)
test  <- dplyr::anti_join(data, train, by = 'id')


#Training models
train_simple_model <- lm(Weight~area, data = train)
train_visual_model1 <- lm(Weight~norm_area, data = train)
train_visual_model2 <- lm(Weight~norm_area + Box, data = train)
train_nongenetic_model <- lm(Weight~norm_area + Sex + Age + Box, data = train)
train_fullmodel_nobox <- lm(Weight~norm_area + Sex * Strain + Age, data = train)
train_full_model <- lm(Weight~norm_area + Sex * Strain + Box + Age, data = train)


ypred <- as.numeric(predict(train_full_model, test))

# plot_rmse <- paste("RMSE =", toString(round(rmse(test$Weight, ypred),3)))
# plot_mae <- paste("MAE =", toString(round(mae(test$Weight, ypred),3)))




test$prediction <- ypred
test$Weight



### Strainwise avg stuff


### Gautam base R code

df <- data.frame(Strain = as.factor(sample(letters, 100, replace = TRUE)), Obs = abs(rnorm(100,0,10)), Pred = abs(rnorm(100,0,10))) #toy data

df

Obs_mean <- tapply(df$Obs, df$Strain, mean)
Obs_sd <- tapply(df$Obs, df$Strain, sd)
Pred_mean <- tapply(df$Pred, df$Strain, mean)
Pred_sd <- tapply(df$Pred, df$Strain, sd)

df_Obs <- data.frame(Strain = names(Obs_mean), Obs_mean = Obs_mean, Obs_sd = Obs_sd)
df_Pred <- data.frame(Strain = names(Pred_mean), Pred_mean = Pred_mean, Pred_sd = Pred_sd)


df_Obs
df_Pred





trimmed_obs_preds <- select(test,c(Strain, Weight, prediction))


Observed_mean <- tapply(trimmed_obs_preds$Weight, trimmed_obs_preds$Strain, mean)
Observed_sd <- tapply(trimmed_obs_preds$Weight, trimmed_obs_preds$Strain, sd)
Predicted_mean <- tapply(trimmed_obs_preds$prediction, trimmed_obs_preds$Strain, mean)
Predicted_sd <- tapply(trimmed_obs_preds$prediction, trimmed_obs_preds$Strain, sd)

df_Observed <- data.frame(Strain = names(Obs_mean), Obs_mean = Obs_mean, Obs_sd = Obs_sd)
df_Predicted <- data.frame(Strain = names(Pred_mean), Pred_mean = Pred_mean, Pred_sd = Pred_sd)

df_Observed
df_Predicted

mean_var_df <- merge(df_Observed, df_Predicted, by="Strain")



### Strainwise average 

modelplot <- mean_var_df %>% 
  ggplot(mapping = aes(x = Obs_mean, y = Pred_mean, color = Strain)) +
  geom_abline(intercept = 0, slope = 1, size = 1, color='black')+
  geom_point(size=2)+
  geom_errorbar(aes(ymin=Pred_mean-Pred_sd, ymax=Pred_mean+Pred_sd), width=.5) +
  geom_errorbarh(aes(xmin=Obs_mean-Obs_sd, xmax=Obs_mean+Obs_sd), height=.5) +
  labs(x='True Weight (g)', y='Predicted Weight (g)')+
  ggtitle("Strain-wise Average")+
  theme_bw()
  # theme(legend.position="none")
  #annotate("text", x = 48, y = 50, angle = 45, label = '45 deg', size = 5)+
  #stat_regline_equation(label.x=10, label.y = 34, aes(label = ..adj.rr.label..)) +
  # stat_regline_equation(label.x=10, label.y = 37, aes(label = ..eq.label..)) +
  # annotate("text", x = 10, y = 39, hjust = 0, label = plot_rmse) +
  # annotate("text", x = 10, y = 41, hjust = 0, label = plot_mae) 

modelplot




# Brian's errorbar style (boxes mean +/- sd)
plot_mean_bars = function(...) {ggplot2::stat_summary(fun.y=mean, fun.ymin=function(x) mean(x)-sd(x), fun.ymax=function(x) mean(x)+sd(x), geom='crossbar', width=0.5, ...=...)}
# Errorbar calculation function
get_errorbar_df = function(df, col1, col2, group_column='Strain') {
  temp_df = aggregate(formula(paste0('cbind(',col1,',',col2,')~',group_column)), data=df, mean)
  #temp_df_sd = aggregate(formula(paste0('cbind(',col1,',',col2,')~',group_column)), data=df, sd)
  temp_df_sd = aggregate(formula(paste0('cbind(',col1,',',col2,')~',group_column)), data=df, function(x) sd(x)/sqrt(length(x)))
  names(temp_df) = c(group_column, 'x', 'y')
  names(temp_df_sd) = c(group_column, 'xsd', 'ysd')
  merge(temp_df, temp_df_sd, by.x=group_column, by.y=group_column)
}


allow_overlap=FALSE
temp = get_errorbar_df(test_metrics, 'grooming_duration_secs_0_first_55m_sum', 'grooming_number_bouts_0_first_55m_sum', 'Strain')
temp$is_wild = temp$Strain %in% wild_strains
temp = temp[order(temp$is_wild, decreasing=!allow_overlap),]
pcor1 = ggplot(data=temp, aes(x=x, y=y, xmin=x-xsd, xmax=x+xsd, ymin=y-ysd, ymax=y+ysd, group=Strain, height=ysd/4, width=xsd/4, col=is_wild))+geom_errorbar()+geom_errorbarh()+geom_point()+annotate(geom='text', x=temp$x, y=temp$y+mean(temp$ysd)/2, label=temp$Strain, color=c('#377eb8', '#e41a1c')[as.numeric(temp$is_wild)+1], check_overlap=!allow_overlap)+theme(legend.position='none')+labs(x='Total Grooming',y='Number Bouts')+scale_color_brewer(palette='Set1', direction=-1)



 