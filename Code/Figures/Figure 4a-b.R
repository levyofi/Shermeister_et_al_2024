library(ggplot2)
library(data.table)
library(dplyr)
library(reshape2)
library(plyr)
library(lubridate)

model_confidence_data<-fread("Data/model_confidence_data.csv")

#round the confidence scores to have only one decimal place 
model_confidence_data$col_con<-round_any(model_confidence_data$color_confidence,1)
model_confidence_data$thermo_con<-round_any(model_confidence_data$thermo_confidence,1)

#create factors
model_confidence_data$wrong_color<-as.factor(model_confidence_data$wrong_color)
model_confidence_data$col_con<-as.factor(model_confidence_data$col_con)
model_confidence_data$wrong_sun.shade<-as.factor(model_confidence_data$wrong_sun.shade)
model_confidence_data$thermo_con<-as.factor(model_confidence_data$thermo_con)


##removing false positive detections and true negatives
## whenever the model defines the thermo label as "negative" it means that it's not a lizard, but an object in the cage that looks similar to a lizard. it helps us locate object that mislead the model.
model_confidence_data<-model_confidence_data[model_confidence_data$FP==0]
model_confidence_data<-model_confidence_data[model_confidence_data$true_negative==0]

#create aggregated data for the color classification model
color_con<-dcast(data = model_confidence_data,formula = col_con~wrong_color)
colnames(color_con)<-c("col_con","correct","wrong")
color_con$correct_percent<-(color_con$correct/(color_con$correct+color_con$wrong)*100)
color_con$wrong_percent<-(color_con$wrong/(color_con$correct+color_con$wrong)*100)
color_con$col_con_numeric<-as.numeric(as.character(color_con$col_con))
color_con = color_con[color_con$col_con_numeric>=50,]

#create aggregated data for the thermoregulation classification model
thermo_con<-dcast(data = model_confidence_data,formula = thermo_con~wrong_sun.shade)
colnames(thermo_con)<-c("thermo_con","correct","wrong")
thermo_con$correct_percent<-(thermo_con$correct/(thermo_con$correct+thermo_con$wrong)*100)
thermo_con$wrong_percent<-(thermo_con$wrong/(thermo_con$correct+thermo_con$wrong)*100)
thermo_con$thermo_con_numeric<-as.numeric(as.character(thermo_con$thermo_con))
thermo_con = thermo_con[thermo_con$thermo_con_numeric>=50,]



color_plot = ggplot(color_con, aes(x=col_con_numeric)) +
  geom_area(aes(y=correct_percent),fill="gray")+
  geom_area(aes(y=correct/38),fill="red")+ #divided by 38 to scaled the data to 0-100
  theme_classic() +
  scale_y_continuous(expand = c(0,0),breaks = seq(0,100,10),name = "Success (%)",limits = c(0, 100), sec.axis = sec_axis(~.*38,name="Number of images",breaks = seq(0,4000,500)))+
  scale_x_continuous(expand = c(0,1),breaks = seq(0,100,5))+
  xlab("Confidence score (%)") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_text(size=20),
        legend.text=element_text(size=18),
        axis.text=element_text(size=12), axis.title=element_text(size=16),
        axis.title.y.right = element_text(margin = margin(t = 0, r = 10, b = 0, l = 10))) +  # Increase distance between right y-axis title and axis
  scale_fill_discrete(name = "Validation results", labels = c("Correct", "Wrong")) +
  geom_vline(xintercept=95, linetype="dotted")


ggsave("color_validation_area_plot.png", plot = color_plot, dpi = 300, width = 6, height = 4, units = "in")

thermo_plot = ggplot(thermo_con, aes(x=thermo_con_numeric)) +
  geom_area(aes(y=correct_percent),fill="gray")+
  geom_area(aes(y=((correct+wrong)/42)),fill="red")+
  theme_classic() +
  scale_y_continuous(expand = c(0,0),breaks = seq(0,100,10),name = "Success (%)",sec.axis = sec_axis(~.*38,name="Number of images",breaks = seq(0,4000,500)))+
  scale_x_continuous(expand = c(0,1),breaks = seq(0,100,5))+
  xlab("Confidence score (%)") +
  theme(plot.title = element_text(hjust = 0.5,size = 20),legend.title=element_text(size=20),
        legend.text=element_text(size=18),
        axis.text=element_text(size=12), axis.title=element_text(size=16),
        axis.title.y.right = element_text(margin = margin(t = 0, r = 10, b = 0, l = 10))) +  # Increase distance between right y-axis title and axis
  scale_fill_discrete(name = "Validation results", labels = c("Correct", "Wrong")) +
  geom_vline(xintercept=95, linetype="dotted")


ggsave("thermo_validation_area_plot.png", plot = thermo_plot, dpi = 300, width = 6, height = 4, units = "in")

## percentage of the images I had after defining 95 confidence level threshold
###thermo
sum(thermo_con$correct[thermo_con$thermo_con_numeric>=95])/sum(thermo_con$correct)*100
###color
sum(color_con$correct[color_con$col_con_numeric>=95])/sum(color_con$correct)*100





