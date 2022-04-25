######################################################################################
# JAX Internship R-Script
# Pixel Analysis Code
# written by Kayla Dixon

# Experiment 1
ellfitdata <- read.csv(file=file.choose())
segdata <- read.csv(file=file.choose())

ellfitlm <- lm(ellfitdata$Weight ~ ellfitdata$`Average Pixel Count`)
plot(ellfitdata$Weight ~ ellfitdata$`Average Pixel Count`)
abline(ellfitlm)
summary(ellfitlm)

seglm <- lm(segdata$Weight ~ segdata$Average.Pixel.Count)
plot(segdata$Weight ~ segdata$Average.Pixel.Count)
abline(seglm)
summary(seglm)

# Experiment 2
e2data <-read.csv(file=file.choose())

mpclm <- lm(e2data$Weight~e2data$Median.Pixel.Count)
summary(mpclm)

wpclm <- lm(e2data$Weight~e2data$Walking.Pixel.Count)
summary(wpclm)

spclm <- lm(e2data$Weight~e2data$Sitting.Pixel.Count)
summary(spclm)

ampclm <- lm(e2data$Weight~e2data$Average.Median.Pixel.Count)
summary(ampclm)

# Finalized Pixel Analysis
setwd("/personal/ksdixo23/JAX Internship Stuff/")
disdata <- read.csv("ellfit_seg_distribution.csv")
disdata$Method <- as.factor(disdata$Method)
disdata$Video <- as.factor(disdata$Video)
disdata_BalbcJ <- disdata[which(disdata$Video=='BalbcJ'),]
disdata_BTBR <- disdata[which(disdata$Video=='BTBR'),]
disdata_LL1_3_S135 <- disdata[which(disdata$Video=='LL1_3_S135'),]
data <- read.csv("BTBR_LL1_4_S145.csv")

## Violin Plots
library("ggplot2")
ggplot(disdata_BalbcJ, aes(x=Method, y=PixelCountPerFrame)) + geom_violin() +
  coord_flip()
ggplot(disdata_BalbcJ, aes(x=Method, y=PixelCountPerFrame)) + geom_violin() + geom_boxplot(width=0.2) +
  coord_flip()
ggplot(disdata_BalbcJ, aes(x=Method, y=PixelCountPerFrame)) + geom_violin() + 
  stat_summary(fun.y=mean, geom="point", color="red")
ggplot(disdata_BalbcJ, aes(x=Method, y=PixelCountPerFrame)) + geom_violin() + geom_boxplot(width=0.2) +
  coord_flip() + stat_summary(fun=mean, geom="point", color="red", size=1)
ggplot(disdata, aes(x=Method, y=PixelCountPerFrame, fill=Video)) + geom_violin() +
  coord_flip()

## Multiple Historgrams on One Graph
ggplot(disdata_BalbcJ, aes(x=PixelCountPerFrame, fill=Method)) +
  geom_histogram(binwidth=100)
ggplot(disdata_BalbcJ, aes(x=PixelCountPerFrame, fill=Method)) +
  geom_histogram(binwidth=100) + facet_grid(Method ~ .)
ggplot(disdata, aes(x=PixelCountPerFrame, fill=Method)) +
  geom_histogram(binwidth=100) + facet_grid(Video ~ .)
ggplot(disdata, aes(x=PixelCountPerFrame, fill=Method)) +
  geom_histogram(binwidth=100) + facet_grid(Video+Method ~ .)
ggplot(disdata_BalbcJ, aes(x=PixelCountPerFrame)) +
  geom_histogram(bins=25) + xlim(0, 5000) + 
  facet_grid(Method ~ .)
ggplot(disdata_BTBR, aes(x=PixelCountPerFrame)) +
  geom_histogram(bins=25) + xlim(0, 5000) + 
  facet_grid(Method ~ .)
ggplot(disdata_LL1_3_S135, aes(x=PixelCountPerFrame)) +
  geom_histogram(bins=25) + xlim(0, 5000) + 
  facet_grid(Method ~ .)
ggplot(data, aes(x=PixelCountPerFrame, fill=Method)) +
  geom_histogram(binwidth=100) + facet_grid(Video+Method ~ .)
