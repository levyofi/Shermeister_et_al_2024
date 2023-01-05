#Combining model results (07.05) with IButtons data and meterological station data
library(data.table)
library(lubridate)
library(tidyr)
library(reshape2)
library(dplyr)
library(ggplot2)
library(plyr)

setwd("D:/OfirL6/Documents/Ben/R/R/github/case study/")

#ibuttons data
ibuttons_data<-fread("all_data_ibuttons_second_year.csv")

#meteorological station data
meteo_data<-fread("all_data_meteo_SECOND_YEAR.csv")

#model results
model_data_07.05<-fread("model_confidence_07.05.ALL_IMAGES_06.12.22.csv")


#remove unnecessary columns and organizing data

model_data_07.05$col_con<-round_any(model_data_07.05$color_confidence,1)
model_data_07.05$thermo_con<-round_any(model_data_07.05$thermo_confidence,1)
model_data_07.05<-subset(model_data_07.05, select = -c(thermo_confidence,color_confidence,count,result) )
colnames(model_data_07.05)[4]<-"time"
#confidence threshold = 95
model_data_07.05<-model_data_07.05[model_data_07.05$col_con>=95 & model_data_07.05$thermo_con>=95]



meteo_data<-subset(meteo_data, select = -c(Timestamp,WindDir,WindLull,WindGust,LightningStrikeCount,HeatIndex,DewPoint,WetBulbTemperature,DeltaT,AirDensity,TimestampEpoch,StationName,date.time,winter_time))
colnames(meteo_data)[c(9,10)]<-c("date","time")
meteo_data<-meteo_data[meteo_data$date=="2022-05-07"]

ibuttons_data<-separate(ibuttons_data, realtime, c("date", "time"), sep = " ",remove = FALSE) 
ibuttons_data<-ibuttons_data[ibuttons_data$date=="2022-05-07"]




model_data_07.05$round.time<-hms(model_data_07.05$time)
model_data_07.05$round.time<-round_date(as.POSIXct.numeric(model_data_07.05$round.time,origin = "2022-11-14",tz="GMT"),"10 minute")
model_data_07.05$round.time<-format(as.POSIXct(model_data_07.05$round.time), format = "%H:%M:%S")

ibuttons_data$round.time<-hms(ibuttons_data$time)
ibuttons_data$round.time<-round_date(as.POSIXct.numeric(ibuttons_data$round.time,origin = "2022-11-14",tz="GMT"),"10 minute")
ibuttons_data$round.time<-format(as.POSIXct(ibuttons_data$round.time), format = "%H:%M:%S")

meteo_data$round.time<-hms(meteo_data$time)
meteo_data$round.time<-round_date(as.POSIXct.numeric(meteo_data$round.time,origin = "2022-11-14",tz="GMT"),"10 minute")
meteo_data$round.time<-format(as.POSIXct(meteo_data$round.time), format = "%H:%M:%S")


#checking differences between ibuttons from different enclosures
ggplot(ibuttons_data, aes(x=round.time, y=temp, color=ibutton, shape = position,
              group=interaction(position, ibutton))) + 
  geom_point() + geom_line()

#spliting the ibuttons position data into two columns (shade,exposed)
#to merge between the dataframes and between the two ibuttons from the same enclosure (exposed & shade)
#I renamed the ibuttons' names so it will not include the different letter fro exposed or shade
# for example: instead of "H1E" it will appear as "H1", and a different column for the shaded ibutton and the exposed one
# I did the same with the enclosure name for the concatenate column in the model dataframe:
#instead of "H1E IMAG0007", I will have another enclosure column with only "H1".

ibuttons_data<-subset(ibuttons_data, select = -c(realtime,date,time))
colnames(ibuttons_data)[2]<-"enclosure"
ibuttons_data$enclosure<-substr(ibuttons_data$enclosure,1,2)
ibuttons_data_splited<-reshape2::dcast(ibuttons_data, ...~position, value.var='temp')
model_data_07.05$enclosure<-substr(model_data_07.05$Concatenate,1,2)

meteo_data<-subset(meteo_data, select = -c(date,time))

#merging the dataframes

model_ibutton<-merge(model_data_07.05,ibuttons_data_splited,by=c("round.time","enclosure"))
model_ibutton_meteo<-merge(model_ibutton,meteo_data[!duplicated(meteo_data[,"round.time"]),],by="round.time")

#population column
model_ibutton_meteo$pop<-substr(model_ibutton_meteo$enclosure,1,1)


#remove unnecessary columns
model_ibutton_meteo<-subset(model_ibutton_meteo, select = -c(remarks,wrong_color,wrong_sun.shade,
non_detected,correct,wrong_both,FP,FN,HARD_TO_DETECT,HARD_TO_DETERMINE_SHADE,HARD_TO_DETERMINE_COLOR,col_con,thermo_con))

colnames(model_ibutton_meteo)[9]<-"shaded"
model_ibutton_meteo$SolarRadiation<-gsub(" W/m2","",model_ibutton_meteo$SolarRadiation)
model_ibutton_meteo$SolarRadiation<-as.numeric(as.character(model_ibutton_meteo$SolarRadiation))
model_ibutton_meteo$AirTemp<-gsub(" C","",model_ibutton_meteo$AirTemp)
model_ibutton_meteo$AirTemp<-as.numeric(as.character(model_ibutton_meteo$AirTemp))

model_ibutton_meteo$exposed.round<-round(model_ibutton_meteo$exposed)
model_ibutton_meteo$shaded.round<-round(model_ibutton_meteo$shaded)

model_ibutton_meteo$ibutton<-substr(model_ibutton_meteo$Concatenate,1,3)


#thermal behavior over time  ####
model_ibutton_meteo$sun<-ifelse(model_ibutton_meteo$thermo_label=="Sun",1,ifelse(model_ibutton_meteo$thermo_label=="Shade",0,"negative"))
model_ibutton_meteo$round.30<-hms(model_ibutton_meteo$time)
model_ibutton_meteo$round.30<-round_date(as.POSIXct.numeric(model_ibutton_meteo$round.30,origin = "2022-11-14",tz="GMT"),"30 minutes")
model_ibutton_meteo$round.30<-format(as.POSIXct(model_ibutton_meteo$round.30), format = "%H:%M:%S")



#Manual validation#####
## Now I will compare the results to the manual validation
manual_07.05<-fread("lizard_detections_validation_07.05.2022_FOR_MODEL.csv")
#organizing the dataframe like the model dataframe
colnames(manual_07.05)[1]<-"thermo_label"
manual_07.05$color[is.na(manual_07.05$color)]<-"NA"
manual_07.05$thermo_label[is.na(manual_07.05$thermo_label)]<-"NA"
manual_07.05<-manual_07.05[,c(3,6,1,2,4)]
manual_07.05$Concatenate<-paste(manual_07.05$camera,manual_07.05$filename)
manual_07.05$round.time<-hms(manual_07.05$time)
manual_07.05$round.time<-round_date(as.POSIXct.numeric(manual_07.05$round.time,origin = "2022-11-14",tz="GMT"),"10 minute")
manual_07.05$round.time<-format(as.POSIXct(manual_07.05$round.time), format = "%H:%M:%S")
manual_07.05$enclosure<-substr(manual_07.05$Concatenate,1,2)
manual_07.05$pop<-substr(manual_07.05$Concatenate,1,1)
manual_07.05<-manual_07.05[,c(7,3,4,5,6,8,9)]
manual_07.05$thermo_label[manual_07.05$thermo_label=="SUN"]<-"Sun"
manual_07.05$thermo_label[manual_07.05$thermo_label=="SHADE"]<-"Shade"

#merging the dataframes
manual_ibutton<-merge(manual_07.05,ibuttons_data_splited,by=c("round.time","enclosure"))
manual_ibutton_meteo<-merge(manual_ibutton,meteo_data[!duplicated(meteo_data[,"round.time"]),],by="round.time")


colnames(manual_ibutton_meteo)[9]<-"shaded"
manual_ibutton_meteo$SolarRadiation<-gsub(" W/m2","",manual_ibutton_meteo$SolarRadiation)
manual_ibutton_meteo$SolarRadiation<-as.numeric(as.character(manual_ibutton_meteo$SolarRadiation))
manual_ibutton_meteo$AirTemp<-gsub(" C","",manual_ibutton_meteo$AirTemp)
manual_ibutton_meteo$AirTemp<-as.numeric(as.character(manual_ibutton_meteo$AirTemp))




manual_ibutton_meteo$sun<-ifelse(manual_ibutton_meteo$thermo_label=="Sun",1,ifelse(manual_ibutton_meteo$thermo_label=="Shade",0,"negative"))
manual_ibutton_meteo$round.30<-hms(manual_ibutton_meteo$time)
manual_ibutton_meteo$round.30<-round_date(as.POSIXct.numeric(manual_ibutton_meteo$round.30,origin = "2022-11-14",tz="GMT"),"30 minutes")
manual_ibutton_meteo$round.30<-format(as.POSIXct(manual_ibutton_meteo$round.30), format = "%H:%M:%S")


# compare the model predictions with the manual validation ####
## compare between populations
### change the pop names to groups 1 and 2
pop.group <- c("Group 1","Group 2")
names(pop.group) <- c("H", "Y")

ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative"], aes( y=SolarRadiation)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA"], aes( y=SolarRadiation, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Solar Radiation") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ pop,labeller = labeller(pop = pop.group)) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())


ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative"], aes( y=shaded)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA"], aes( y=shaded, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Temperature in the shade") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ pop,labeller = labeller(pop = pop.group)) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())



ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative"], aes( y=exposed)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA"], aes( y=exposed, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Temperature in the sun") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ pop,labeller = labeller(pop = pop.group)) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())


##compare between the colors
###change the colors names to be the same in the model as in the manual

model_ibutton_meteo$color_label<-tolower(model_ibutton_meteo$color_label)
colnames(manual_ibutton_meteo)[4]<-"color_label"
manual_ibutton_meteo$color_label<-tolower(manual_ibutton_meteo$color_label)


ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative" & model_ibutton_meteo$color_label!="negative"], aes( y=SolarRadiation)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA" & manual_ibutton_meteo$color_label!="na"], aes( y=SolarRadiation, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Solar Radiation") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ color_label) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())

ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative" & model_ibutton_meteo$color_label!="negative"], aes( y=shaded)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA" & manual_ibutton_meteo$color_label!="na"], aes( y=shaded, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Temperature in the shade") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ color_label) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())

ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative" & model_ibutton_meteo$color_label!="negative"], aes( y=exposed)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA" & manual_ibutton_meteo$color_label!="na"], aes( y=exposed, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Temperature in the sun") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ color_label) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())


## compare between morning noon and afternoon-evening. (07-11,11-15,15-19)
model_ibutton_meteo$time_of_day<-ifelse(hms(model_ibutton_meteo$round.time)<hms("11:00:00"),"Morning",ifelse(hms(model_ibutton_meteo$round.time)>=hms("11:00:00") & hms(model_ibutton_meteo$round.time)<hms("15:00:00"),"Noon","Afternoon"))
manual_ibutton_meteo$time_of_day<-ifelse(hms(manual_ibutton_meteo$round.time)<hms("11:00:00"),"Morning",ifelse(hms(manual_ibutton_meteo$round.time)>=hms("11:00:00") & hms(manual_ibutton_meteo$round.time)<hms("15:00:00"),"Noon","Afternoon"))



ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative"], aes( y=SolarRadiation)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA"], aes( y=SolarRadiation, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Solar Radiation") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ factor(time_of_day,levels=c('Morning','Noon','Afternoon'))) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())

ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative"], aes( y=shaded)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA"], aes( y=shaded, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Temperature in the shade") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ factor(time_of_day,levels=c('Morning','Noon','Afternoon'))) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())

ggplot(model_ibutton_meteo[model_ibutton_meteo$thermo_label!="negative"], aes( y=exposed)) + 
  geom_boxplot(aes(fill="\nmodel\nprediction\n"), width = 0.25, position= position_nudge(x=-.2)) +
  geom_boxplot(data = manual_ibutton_meteo[manual_ibutton_meteo$thermo_label!="NA"], aes( y=exposed, fill="\nmanual\nvalidation\n"), width = 0.25, position= position_nudge(x=.1)) +
  scale_fill_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Temperature in the sun") +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  facet_grid(thermo_label ~ factor(time_of_day,levels=c('Morning','Noon','Afternoon'))) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())





activity_over_time_model_both_pop <- model_ibutton_meteo[model_ibutton_meteo$sun!="negative"] %>%
  group_by(round.30) %>%
  dplyr::summarise(plyr::count(sun),exposed=mean(exposed),shaded=mean(shaded),radiation=mean(SolarRadiation))

activity_over_time_model_both_pop<-reshape2::dcast(activity_over_time_model_both_pop, ...~x, value.var='freq')
colnames(activity_over_time_model_both_pop)[c(5,6)]<-c("shade","sun")
activity_over_time_model_both_pop$shade[is.na(activity_over_time_model_both_pop$shade)]<-0
activity_over_time_model_both_pop$sun[is.na(activity_over_time_model_both_pop$sun)]<-0
activity_over_time_model_both_pop$percent_sun<-(activity_over_time_model_both_pop$sun/(activity_over_time_model_both_pop$sun+activity_over_time_model_both_pop$shade)*100)
activity_over_time_model_both_pop$percent_shade<-(activity_over_time_model_both_pop$shade/(activity_over_time_model_both_pop$sun+activity_over_time_model_both_pop$shade)*100)


activity_over_time_manual_both_pop <- manual_ibutton_meteo[manual_ibutton_meteo$sun!="negative"] %>%
  group_by(round.30) %>%
  dplyr::summarise(plyr::count(sun),exposed=mean(exposed),shaded=mean(shaded),radiation=mean(SolarRadiation))

activity_over_time_manual_both_pop<-reshape2::dcast(activity_over_time_manual_both_pop, ...~x, value.var='freq')
colnames(activity_over_time_manual_both_pop)[c(5,6)]<-c("shade","sun")
activity_over_time_manual_both_pop$shade[is.na(activity_over_time_manual_both_pop$shade)]<-0
activity_over_time_manual_both_pop$sun[is.na(activity_over_time_manual_both_pop$sun)]<-0
activity_over_time_manual_both_pop$percent_sun<-(activity_over_time_manual_both_pop$sun/(activity_over_time_manual_both_pop$sun+activity_over_time_manual_both_pop$shade)*100)
activity_over_time_manual_both_pop$percent_shade<-(activity_over_time_manual_both_pop$shade/(activity_over_time_manual_both_pop$sun+activity_over_time_manual_both_pop$shade)*100)


activity_over_time_model_both_pop$hm30<- format(as.POSIXct(activity_over_time_model_both_pop$round.30, format="%H:%M:%S"), "%H:%M")
activity_over_time_manual_both_pop$hm30<- format(as.POSIXct(activity_over_time_manual_both_pop$round.30, format="%H:%M:%S"), "%H:%M")


#organizing the relevant time for the x axis
vec <- activity_over_time_model_both_pop$hm30
x_axis_time<-as.vector(NULL)
count = 0
# looping over the vector elements
for (i in vec){
  
  # incrementing count
  count= count + 1
  # checking if count is equal to third
  # element
  if(count %% 2 == 0){
    x_axis_time<- append(x_axis_time,i)
  }  
}

ggplot(activity_over_time_model_both_pop, aes(x=hm30)) +
  geom_line(aes(y=percent_sun,group="\nmodel\nprediction\n",color="\nmodel\nprediction\n"),size=2)+
  geom_line(data=activity_over_time_manual_both_pop,aes(y=(percent_sun),group="\nmanual\nvalidation\n",color="\nmanual\nvalidation\n"),size=2) +
  scale_color_manual(values=c("skyblue3", "lightblue1"),breaks=c('\nmodel\nprediction\n', '\nmanual\nvalidation\n'))+
  ylab("Activity in the Sun (%)") +
  xlab("Time") +
  scale_x_discrete(breaks=x_axis_time) +
  theme_bw() +
  ggtitle("Model prediction VS Manual validation") + 
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.title=element_text(size=18),axis.text = element_text(size = 15))




