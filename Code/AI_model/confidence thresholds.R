
# Confidence thresholds
## To check the best model's confidence threshold I wanted to see the confidence levels of the classifications.
## In order to do so, I need to merge between the dataframe of the manual validation, with all the model's mistakes,
## With the dataframe of the model where I can see the confidence level of each cllasification.
## In pictures with more than one detection (both for the model and the manual validation) I will need to check it manualy.
## But to save some time I will create a dataframe of pictures that both the model and the manual validation detected only one lizard (bacause than I can merge between the dataframes and for each filename there will be one row only)

library(ggplot2)
library(data.table)
library(dplyr)
library(reshape2)
library(plyr)
library(lubridate)

setwd("~/Ben/enclosures_exp/encexp/")


validation_07.05_NEW<-read.csv("lizard_detections_validation_07.05.2022_FOR_MODEL.csv")
colnames(validation_07.05_NEW)[1] <- 'sun.shade'
validation_07.05_NEW$color[is.na(validation_07.05_NEW$color)]<-"NA"

validation_07.05_NEW_result.count <- validation_07.05_NEW %>%
  group_by(model_result) %>%
  dplyr:: summarise(count = n()) %>%
  top_n(n = 5, wt = count)
validation_07.05_NEW_result.count$percent<-(validation_07.05_NEW_result.count$count/sum(validation_07.05_NEW_result.count$count)*100)
validation_07.05_NEW_result.count$date<-"07.05"



validation_07.05_NEW$Concatenate<-paste(validation_07.05_NEW$camera,validation_07.05_NEW$filename)

validation_07.05_NEW <- validation_07.05_NEW %>%
  group_by(Concatenate) %>%
  dplyr:: mutate(count = n())
validation_07.05_result.single_count<-validation_07.05_NEW[validation_07.05_NEW$non_detected==0 & validation_07.05_NEW$count==1,]
validation_07.05_result.multiple_count<-validation_07.05_NEW[validation_07.05_NEW$non_detected==0 & validation_07.05_NEW$count>1,]


model_07.05<-read.csv("model_result_august_07.05_REMARKS.csv")
model_07.05$Concatenate<-paste(model_07.05$camera,model_07.05$filename)

model_07.05 <- model_07.05 %>%
  group_by(Concatenate) %>%
  dplyr:: mutate(count = n())
model_07.05_result.single_count<-model_07.05[model_07.05$fp==0 & model_07.05$count==1,]
model_07.05_result.multiple_count<-model_07.05[model_07.05$fp==0 & model_07.05$count>1,]



##pictures with only one detection both in the manual validation and in the model


model_and_validation_result_07.05.single_count<-merge(validation_07.05_result.single_count,model_07.05_result.single_count,by="Concatenate",all = T)
model_and_validation_result_07.05.single_count<-model_and_validation_result_07.05.single_count[is.na(model_and_validation_result_07.05.single_count$count.x)==F & is.na(model_and_validation_result_07.05.single_count$count.y)==F,]
model_and_validation_result_07.05.single_count<-subset(model_and_validation_result_07.05.single_count, select = -c(temp,X,xmin,ymin,xmax,ymax) )

model_and_validation_result_07.05.single_count$col_con<-round_any(model_and_validation_result_07.05.single_count$color_confidence,1)
model_and_validation_result_07.05.single_count$thermo_con<-round_any(model_and_validation_result_07.05.single_count$thermo_confidence,1)


write.csv(model_and_validation_result_07.05.single_count, file = paste0("D:/OfirL6/Documents/Ben/enclosures_exp/encexp/model_and_validation_result_07.05.single_count_",format(today(), "%d.%m.%y"),".csv"), row.names = FALSE)



model_and_validation_result_07.05.single_count$wrong_color<-as.factor(model_and_validation_result_07.05.single_count$wrong_color)
model_and_validation_result_07.05.single_count$col_con<-as.factor(model_and_validation_result_07.05.single_count$col_con)
model_and_validation_result_07.05.single_count$wrong_sun.shade<-as.factor(model_and_validation_result_07.05.single_count$wrong_sun.shade)
model_and_validation_result_07.05.single_count$thermo_con<-as.factor(model_and_validation_result_07.05.single_count$thermo_con)


ggplot(model_and_validation_result_07.05.single_count, aes(x=col_con, y=wrong_color,fill=wrong_color)) +
  geom_bar(stat="identity")
ggplot(model_and_validation_result_07.05.single_count, aes(x=thermo_con, y=wrong_sun.shade,fill=wrong_sun.shade)) +
  geom_bar(stat="identity")

color_con<-dcast(data = model_and_validation_result_07.05.single_count,formula = col_con~wrong_color)
colnames(color_con)<-c("col_con","correct","wrong")
color_con$correct_percent<-(color_con$correct/(color_con$correct+color_con$wrong)*100)
color_con$wrong_percent<-(color_con$wrong/(color_con$correct+color_con$wrong)*100)


color_con_long<-melt(color_con[c(1,4,5)],id.vars ="col_con" )
color_con_long$col_con_numeric<-as.numeric(as.character(color_con_long$col_con))

ggplot(color_con_long, aes(x=col_con_numeric, y=value,fill=variable)) +
  geom_bar(stat="identity")+
  ylab("success (%)") +
  xlab("Confidence level") +
  theme_classic() +
  scale_y_continuous(expand = c(0,0))+
  scale_x_continuous(expand = c(0,0))+
  ggtitle("model confidence for color classification") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_text(size=20),
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18))+
  scale_fill_discrete(name = "Validation results", labels = c("Correct", "Wrong"))




thermo_con<-dcast(data = model_and_validation_result_07.05.single_count,formula = thermo_con~wrong_sun.shade)
colnames(thermo_con)<-c("thermo_con","correct","wrong")
thermo_con$correct_percent<-(thermo_con$correct/(thermo_con$correct+thermo_con$wrong)*100)
thermo_con$wrong_percent<-(thermo_con$wrong/(thermo_con$correct+thermo_con$wrong)*100)


thermo_con_long<-melt(thermo_con[c(1,4,5)],id.vars ="thermo_con" )
thermo_con_long$thermo_con_numeric<-as.numeric(as.character(thermo_con_long$thermo_con))

ggplot(thermo_con_long, aes(x=thermo_con, y=value,fill=variable)) +
  geom_bar(stat="identity") +
  ylab("Percentage") +
  xlab("Confidence level") +
  theme_classic() +
  ggtitle("model confidence for thermoregulation behavior classification") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=10), axis.title=element_text(size=18))

ggplot(thermo_con_long, aes(x=thermo_con_numeric, y=value,fill=variable)) +
  geom_bar(stat="identity") +
  ylab("success (%)") +
  xlab("Confidence level") +
  theme_classic() +
  scale_y_continuous(expand = c(0,0))+
  scale_x_continuous(expand = c(0,0))+
  ggtitle("model confidence for thermoregulation behavior classification") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_text(size=20),
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18))+
  scale_fill_discrete(name = "Validation results", labels = c("Correct", "Wrong"))


## To analyze all the other results - the pictures with more than one detection
##in either the model or the human validation
## I will create dataframes that doesn't include the pictures with one detection

model_result_07.05.multiple_count<-model_07.05[!(model_07.05$Concatenate %in% model_and_validation_result_07.05.single_count$Concatenate),]
validation_07.05_result.multiple_count<-validation_07.05_NEW[!(validation_07.05_NEW$Concatenate %in% model_and_validation_result_07.05.single_count$Concatenate),]


write.csv(model_result_07.05.multiple_count, file = paste0("D:/OfirL6/Documents/Ben/enclosures_exp/encexp/model_result_07.05.multiple_count_",format(today(), "%d.%m.%y"),".csv"), row.names = FALSE)
write.csv(validation_07.05_result.multiple_count, file = paste0("D:/OfirL6/Documents/Ben/enclosures_exp/encexp/validation_07.05_result.multiple_count_",format(today(), "%d.%m.%y"),".csv"), row.names = FALSE)




## Confidence threshold ALL INAGES ####
## After I organized all the multiple detections data and merged them with the single detections data, using excel
## I will run the confidence code on the new dataframe


model_confidence_07.05.ALL_IMAGES<-fread("model_confidence_07.05.ALL_IMAGES_06.12.22.csv")


model_confidence_07.05.ALL_IMAGES$col_con<-round_any(model_confidence_07.05.ALL_IMAGES$color_confidence,1)
model_confidence_07.05.ALL_IMAGES$thermo_con<-round_any(model_confidence_07.05.ALL_IMAGES$thermo_confidence,1)



model_confidence_07.05.ALL_IMAGES$wrong_color<-as.factor(model_confidence_07.05.ALL_IMAGES$wrong_color)
model_confidence_07.05.ALL_IMAGES$col_con<-as.factor(model_confidence_07.05.ALL_IMAGES$col_con)
model_confidence_07.05.ALL_IMAGES$wrong_sun.shade<-as.factor(model_confidence_07.05.ALL_IMAGES$wrong_sun.shade)
model_confidence_07.05.ALL_IMAGES$thermo_con<-as.factor(model_confidence_07.05.ALL_IMAGES$thermo_con)


##removing false positive detections and true negatives
## whenever the model defines the thermo label as "negative" it means that it's not a lizard, but an object in the cage that looks similar to a lizard. it helps us locate object that mislead the model.
model_confidence_07.05.NO_FP<-model_confidence_07.05.ALL_IMAGES[model_confidence_07.05.ALL_IMAGES$FP==0]
model_confidence_07.05.NO_FP<-model_confidence_07.05.NO_FP[model_confidence_07.05.NO_FP$true_negative==0]

color_con<-dcast(data = model_confidence_07.05.NO_FP,formula = col_con~wrong_color)
colnames(color_con)<-c("col_con","correct","wrong")
color_con$correct_percent<-(color_con$correct/(color_con$correct+color_con$wrong)*100)
color_con$wrong_percent<-(color_con$wrong/(color_con$correct+color_con$wrong)*100)
color_con$col_con_numeric<-as.numeric(as.character(color_con$col_con))


thermo_con<-dcast(data = model_confidence_07.05.NO_FP,formula = thermo_con~wrong_sun.shade)
colnames(thermo_con)<-c("thermo_con","correct","wrong")
thermo_con$correct_percent<-(thermo_con$correct/(thermo_con$correct+thermo_con$wrong)*100)
thermo_con$wrong_percent<-(thermo_con$wrong/(thermo_con$correct+thermo_con$wrong)*100)
thermo_con$thermo_con_numeric<-as.numeric(as.character(thermo_con$thermo_con))


##  combining success percentages with the number of images for each confidence level


ggplot(color_con[14:NROW(color_con),], aes(x=col_con_numeric)) +
  geom_area(aes(y=correct_percent),fill="#b4cbd4")+
  geom_area(aes(y=correct/38),fill="#42659A")+
  theme_classic() +
  scale_y_continuous(expand = c(0,0),breaks = seq(0,100,5),name = "Success (%)",sec.axis = sec_axis(~.*38,name="Number of images",breaks = seq(0,4000,500)))+
  scale_x_continuous(expand = c(0,1),breaks = seq(0,100,5))+
  xlab("Confidence level") +
  ggtitle("Model confidence for color classification") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_text(size=20),
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18))+
  scale_fill_discrete(name = "Validation results", labels = c("Correct", "Wrong")) +
  geom_hline(yintercept=80)



ggplot(thermo_con[6:NROW(thermo_con),], aes(x=thermo_con_numeric)) +
  geom_area(aes(y=correct_percent),fill="#b4cbd4")+
  geom_area(aes(y=((correct+wrong)/42)),fill="#42659A")+
  theme_classic() +
  scale_y_continuous(expand = c(0,0),breaks = seq(0,100,5),name = "Success (%)",sec.axis = sec_axis(~.*42,name="Number of images",breaks = seq(0,4000,500)))+
  scale_x_continuous(expand = c(0,1),breaks = seq(0,100,5))+
  xlab("Confidence level") +
  ggtitle("Model confidence for thermoregulation behavior classification") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_text(size=20),
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18))+
  scale_fill_discrete(name = "Validation results", labels = c("Correct", "Wrong")) +
  geom_hline(yintercept=80)



## percentage of the images I had after defining 95 confidence level threshold
###thermo
sum(thermo_con$correct[thermo_con$thermo_con_numeric>=95])/sum(thermo_con$correct)*100
###color
sum(color_con$correct[color_con$col_con_numeric>=95])/sum(color_con$correct)*100






