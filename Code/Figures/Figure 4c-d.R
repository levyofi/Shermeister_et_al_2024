library(data.table)
library(caret)
library(lubridate)
library(ggplot2)
library(yardstick)
library(ggplot2)
library(gridExtra)

data<-fread("model_confidence_07.05.ALL_IMAGES_29.01.23.csv")
data<-data[data$FP==0]
colnames(data)[c(2,4,6)]=c("model_thermo","model_color","time")
data=data[data$model_thermo!="negative" & data$true_thermo!="negative" & data$true_thermo!="unclear",]
data$model_color[data$model_color=="negative"]="unclear"
data$model_thermo=tolower(data$model_thermo)
data$model_color=tolower(data$model_color)

data$true_thermo=as.factor(data$true_thermo)
data$model_thermo=as.factor(data$model_thermo)
data$true_color=as.factor(data$true_color)
data$model_color=as.factor(data$model_color)

#95 confidence score

#confusion matrix for thermo label with confidence score above 95
thermo_95=data[data$thermo_confidence>=95,]
matrix_thermo_95<-confusionMatrix(thermo_95$model_thermo, thermo_95$true_thermo)
print(matrix_thermo_95$table)
mat_tab<- matrix(matrix_thermo_95$table)


#Ploting confusion matrix - 
#Becaues the numbers of detections is not equal for all cases (e.g. there are much more cases of lizards in the sun),
#I want to create a plot with the percentages of each case and not only the actual numbers.
cm_thermo_95 <- conf_mat(thermo_95,true_thermo, model_thermo)
#create a dataframe with the percentages of each case within each label
cm_thermo_95_precent=cm_thermo_95
cm_thermo_95_precent$table[1]=(cm_thermo_95$table[1])/(cm_thermo_95$table[1]+cm_thermo_95$table[2])*100
cm_thermo_95_precent$table[2]=(cm_thermo_95$table[2])/(cm_thermo_95$table[1]+cm_thermo_95$table[2])*100
cm_thermo_95_precent$table[3]=(cm_thermo_95$table[3])/(cm_thermo_95$table[3]+cm_thermo_95$table[4])*100
cm_thermo_95_precent$table[4]=(cm_thermo_95$table[4])/(cm_thermo_95$table[3]+cm_thermo_95$table[4])*100
#convert the confusion matrix into dataframe so i could add labels to the plot
cm_thermo_95_dataframe=as.data.frame(cm_thermo_95$table)

#plot
#convert the "conf_mat" objects to dataframes
cm_thermo_95_precent_df <- as.data.frame(cm_thermo_95_precent$table)


ggplot(cm_thermo_95_precent_df, aes(Truth,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=paste0(round(Freq,digits = 3),"%")),size=6) +
  scale_fill_gradient(limits = c(0, 100),low="#FFE5B4",high = "#FF851B",na.value = "white", breaks = seq(0, 100, 25)) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),legend.key.width=unit(1.3,"cm"),legend.position = c(0.7,1.01),legend.direction = "horizontal", legend.background = element_blank(),legend.text = element_text(size = 15),legend.title=element_blank(),
        axis.ticks = element_blank(),axis.text=element_text(size=18), axis.title.x=element_text(size=20),axis.title.y = element_text(size=20)) +
  ggtitle("") + 
  theme(plot.title = element_text(size = 20,vjust = -1)) +
  geom_text(data = cm_thermo_95_dataframe,aes(label=paste0("(",Freq,")"),fill=NULL), vjust = 2.4,size=6) # add labels for the frequency values


#confusion matrix for color label with confidence score above 95
color_95=data[data$color_confidence>=95,]
matrix_color_95<-confusionMatrix(color_95$model_color, color_95$true_color)
print(matrix_color_95$table)
mat_tab<- matrix(matrix_color_95$table)

#ploting confusion matrix
cm_color_95 <- conf_mat(color_95,true_color, model_color)

#create a dataframe with the percentages of each case within each label
cm_color_95_precent=cm_color_95
cm_color_95_precent$table[1]=(cm_color_95$table[1])/(sum(cm_color_95$table[1:4]))*100
cm_color_95_precent$table[2]=(cm_color_95$table[2])/(sum(cm_color_95$table[1:4]))*100
cm_color_95_precent$table[3]=(cm_color_95$table[3])/(sum(cm_color_95$table[1:4]))*100
cm_color_95_precent$table[4]=(cm_color_95$table[4])/(sum(cm_color_95$table[1:4]))*100
cm_color_95_precent$table[5]=(cm_color_95$table[5])/(sum(cm_color_95$table[5:8]))*100
cm_color_95_precent$table[6]=(cm_color_95$table[6])/(sum(cm_color_95$table[5:8]))*100
cm_color_95_precent$table[7]=(cm_color_95$table[7])/(sum(cm_color_95$table[5:8]))*100
cm_color_95_precent$table[8]=(cm_color_95$table[8])/(sum(cm_color_95$table[5:8]))*100
cm_color_95_precent$table[9]=(cm_color_95$table[9])/(sum(cm_color_95$table[9:12]))*100
cm_color_95_precent$table[10]=(cm_color_95$table[10])/(sum(cm_color_95$table[9:12]))*100
cm_color_95_precent$table[11]=(cm_color_95$table[11])/(sum(cm_color_95$table[9:12]))*100
cm_color_95_precent$table[12]=(cm_color_95$table[12])/(sum(cm_color_95$table[9:12]))*100
cm_color_95_precent$table[13]=(cm_color_95$table[13])/(sum(cm_color_95$table[13:16]))*100
cm_color_95_precent$table[14]=(cm_color_95$table[14])/(sum(cm_color_95$table[13:16]))*100
cm_color_95_precent$table[15]=(cm_color_95$table[15])/(sum(cm_color_95$table[13:16]))*100
cm_color_95_precent$table[16]=(cm_color_95$table[16])/(sum(cm_color_95$table[13:16]))*100
#convert the confusion matrix into dataframe so i could add labels to the plot
cm_color_95_dataframe=as.data.frame(cm_color_95$table)

#plot
#convert the "conf_mat" objects to dataframes
cm_color_95_precent_df <- as.data.frame(cm_color_95_precent$table)

ggplot(cm_color_95_precent_df, aes(Truth,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=paste0(round(Freq,digits = 3),"%")),size=6) +
  scale_fill_gradient(limits = c(0, 100),low="#FFE5B4",high = "#FF851B",na.value = "white", breaks = seq(0, 100, 25)) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),legend.key.width=unit(1.3,"cm"),legend.position = c(0.7,1.04),legend.direction = "horizontal", legend.background = element_blank(),legend.text = element_text(size = 15),legend.title=element_blank(),
        axis.ticks = element_blank(),axis.text=element_text(size=18), axis.title.x=element_text(size=20),axis.title.y = element_text(size=20)) +
  ggtitle("\n ") + 
  theme(plot.title = element_text(size = 20,vjust = -1)) +
  geom_text(data = cm_color_95_dataframe,aes(label=paste0("(",Freq,")"),fill=NULL), vjust = 2.4,size=6) # add labels for the frequency values
