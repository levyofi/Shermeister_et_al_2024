library(data.table)
library(caret)
library(ggplot2)
library(yardstick)
library(gridExtra)

#Confusion matrix for the no color detection and classification
data<-fread("No color detection and classification results.csv")

#detection model
true_detections = sum(data$true_detections)
false_negatives = sum(data$false_negative)
false_positives = sum(data$false_positive)
# Calculate accuracy
detection_accuracy <- true_detections / (true_detections + false_negatives + false_positives)

# Calculate precision
detection_precision <- true_detections / (true_detections + false_positives)

# Calculate recall (or sensitivity)
detection_recall <- true_detections / (true_detections + false_negatives)

# Calculate F1 score
detection_f1_score <- 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)

# Print the results
cat("detection model Accuracy:", detection_accuracy, "\n")
cat("detection model Recall:", detection_recall, "\n")
cat("detection model Precision:", detection_precision, "\n")
cat("detection model F1 Score:", detection_f1_score, "\n")

#confusion matrix for the thermo classification model
#read the data again
data<-fread("No color detection and classification results.csv")

#confusion matrix for thermo label
conf_thermo$true_thermo=as.factor(conf_thermo$true_thermo)
conf_thermo$model_thermo=as.factor(conf_thermo$model_thermo)
matrix_thermo<-confusionMatrix(conf_thermo$model_thermo, conf_thermo$true_thermo)
print(matrix_thermo$table)
mat_tab<- matrix(matrix_thermo$table)

#Ploting confusion matrix - 
#Becaues the numbers of detections is not equal for all cases (e.g. there are much more cases of lizards in the sun),
#I want to create a plot with the percentages of each case and not only the actual numbers.
cm_thermo <- conf_mat(conf_thermo,true_thermo, model_thermo)
#create a conf_thermoframe with the percentages of each case within each label
cm_thermo_precent=cm_thermo
cm_thermo_precent$table[1]=(cm_thermo$table[1])/(cm_thermo$table[1]+cm_thermo$table[2])*100
cm_thermo_precent$table[2]=(cm_thermo$table[2])/(cm_thermo$table[1]+cm_thermo$table[2])*100
cm_thermo_precent$table[3]=(cm_thermo$table[3])/(cm_thermo$table[3]+cm_thermo$table[4])*100
cm_thermo_precent$table[4]=(cm_thermo$table[4])/(cm_thermo$table[3]+cm_thermo$table[4])*100
#convert the confusion matrix into conf_thermoframe so i could add labels to the plot
cm_thermo_count_df=as.data.frame(cm_thermo$table)
cm_thermo_precent_df = cm_thermo_count_df
cm_thermo_precent_df$Freq[1] = 100*cm_thermo_count_df$Freq[1]/sum(cm_thermo_count_df$Freq[1]+cm_thermo_count_df$Freq[2])
cm_thermo_precent_df$Freq[2] = 100*cm_thermo_count_df$Freq[2]/sum(cm_thermo_count_df$Freq[1]+cm_thermo_count_df$Freq[2])
cm_thermo_precent_df$Freq[3] = 100*cm_thermo_count_df$Freq[3]/sum(cm_thermo_count_df$Freq[3]+cm_thermo_count_df$Freq[4])
cm_thermo_precent_df$Freq[4] = 100*cm_thermo_count_df$Freq[4]/sum(cm_thermo_count_df$Freq[3]+cm_thermo_count_df$Freq[4])

plot = ggplot(cm_thermo_precent_df, aes(Truth,Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=paste0(round(Freq,digits = 3),"%")),size=6) +
  scale_fill_gradient(limits = c(0, 100),low="#FFE5B4",high = "#FF851B",na.value = "white", breaks = seq(0, 100, 25)) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),legend.key.width=unit(1.3,"cm"),legend.position = c(0.7,1.01),legend.direction = "horizontal", legend.background = element_blank(),legend.text = element_text(size = 15),legend.title=element_blank(),
        axis.ticks = element_blank(),axis.text=element_text(size=18), axis.title.x=element_text(size=20),axis.title.y = element_text(size=20)) +
  ggtitle("") + 
  theme(plot.title = element_text(size = 20,vjust = -1)) +
  geom_text(data = cm_thermo_count_df,aes(label=paste0("(",Freq,")"),fill=NULL), vjust = 2.4,size=6) # add labels for the frequency values

plot
ggsave(paste0("Figure 7.jpg"), plot, width = 8, height = 6, dpi = 300)

# precision, recall and F1 for thermo label
classes_thermo<- matrix_thermo$byClass
#classes<- subset(classes, subset=1, select= c('Precision','Recall','F1'))
classes_thermo=classes_thermo[c(5,6,7, 11)]
classes_thermo
