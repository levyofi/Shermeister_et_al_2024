library(stringr)
library(lubridate)
library(insol)
library(brms)
library(ggplot2)
data = read.csv("~/Desktop/University/Masters/Lab/enclosure exp/model_confidence_07.05.ALL_IMAGES_29.01.23.csv", header = T)

data$true_thermo = as.factor(data$true_thermo)

data$true_color = as.factor(data$true_color)
#get enclosure, population and camera from the concatenate id
data$camera=as.factor(substr(data$Concatenate, 1, 3) )
data$enclosure = as.factor(substr(data$camera, 1, 2) )
data$population =  as.factor(substr(data$camera, 1, 1) )

#round to the nearest hour
colnames(data)[6]="time"
hour = hms(data$time)
hour <-round_date(as.POSIXct.numeric(hour,origin = "2022-11-14",tz="GMT"),"hour")
data$hour = lubridate:: hour(hms(format(as.POSIXct(hour), format = "%H:%M:%S")))
data$hour_std = (data$hour - mean(data$hour))/(2*sd(data$hour)) 
str(data)

#datetime
data$datetime<-hms(data$time)
#convert to UTC
data$datetime<-data$datetime-hms("02:00:00")
#add date
data$datetime <-round_date(as.POSIXct.numeric(data$datetime,origin = "2022-05-07",tz="GMT"),"hour")
data$datetime<-ymd_hms(data$datetime)

#calculate absolute julian day in seconds
jd = JD(data$datetime) #number of seconds since the beginning of 1970 (in the UTC timezone)


#calculate solar zenith and azimuth angles

sunv = sunvector(jd,32.11161886503463, 34.80780670958281, timezone = 0) #we took timezone into account already
zenith = sunpos(sunv)[,2]
data$zenith=zenith
data$zenith_std = (data$zenith - mean(data$zenith))/(2*sd(data$zenith)) 

#parts of the day
data$part_of_day<-ifelse(hms(data$time)<hms("11:00:00"),"Morning",ifelse(hms(data$time)>=hms("11:00:00") & hms(data$time)<hms("15:00:00"),"Noon","Afternoon"))
data$part_of_day<-as.factor(data$part_of_day)


#thermoregulation model 
data_thermo = data[data$true_thermo!="unclear" & data$true_thermo!="negative",]
#-------------------------------------------------------------------------------------------------------#
# To save time here is the saved bayesian statistics results of the thermoregulation model
#mod_thermo=readRDS("~/Desktop/University/Masters/Lab/enclosure exp/mod_thermo.rds")
#-------------------------------------------------------------------------------------------------------#
# mod_thermo = brms::brm(wrong_sun.shade~part_of_day*true_color*true_thermo+ (1|enclosure/camera),data = data_thermo,family = 'bernoulli',prior = set_prior('normal(0, 3)'),
#                        iter = 1000,
#                        chains = 2,
#                        cores = 2)

summary(mod_thermo)
# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept                                              -1.65      0.52    -2.66    -0.60 1.00      576      540
# part_of_dayMorning                                     -1.65      0.62    -2.97    -0.54 1.00      948      604
# part_of_dayNoon                                        -0.56      0.32    -1.19     0.08 1.00      773      700
# true_colorred                                           0.31      0.38    -0.46     1.01 1.00      969      834
# true_colorunclear                                      -2.60      1.49    -5.88     0.00 1.00     1080      555
# true_colorwhite                                        -2.28      0.60    -3.44    -1.19 1.00      790      639
# true_thermosun                                         -1.49      0.39    -2.30    -0.73 1.00      700      583
# part_of_dayMorning:true_colorred                        0.88      0.93    -0.97     2.58 1.00      914      626
# part_of_dayNoon:true_colorred                          -0.42      0.45    -1.32     0.46 1.00      861      876
# part_of_dayMorning:true_colorunclear                   -1.00      2.13    -5.39     2.78 1.00     1330      687
# part_of_dayNoon:true_colorunclear                       1.24      1.60    -1.66     4.82 1.00     1038      501
# part_of_dayMorning:true_colorwhite                     -0.84      1.78    -4.62     2.20 1.00      853      662
# part_of_dayNoon:true_colorwhite                         2.38      0.63     1.25     3.71 1.00      747      705
# part_of_dayMorning:true_thermosun                       2.20      0.68     0.93     3.60 1.00      851      728
# part_of_dayNoon:true_thermosun                          0.51      0.44    -0.33     1.37 1.00      681      743
# true_colorred:true_thermosun                           -2.61      1.03    -4.72    -0.67 1.00      740      705
# true_colorunclear:true_thermosun                        0.93      1.95    -3.08     4.41 1.00      916      725
# true_colorwhite:true_thermosun                          0.15      0.85    -1.48     1.72 1.01      708      790
# part_of_dayMorning:true_colorred:true_thermosun         1.11      1.30    -1.31     3.78 1.00      701      664
# part_of_dayNoon:true_colorred:true_thermosun            3.55      1.06     1.62     5.70 1.00      795      705
# part_of_dayMorning:true_colorunclear:true_thermosun     0.03      2.35    -4.42     5.12 1.00     1038      681
# part_of_dayNoon:true_colorunclear:true_thermosun        0.96      1.97    -2.79     5.02 1.00      905      566
# part_of_dayMorning:true_colorwhite:true_thermosun       0.77      1.77    -2.44     4.56 1.01      796      601
# part_of_dayNoon:true_colorwhite:true_thermosun         -1.38      0.88    -3.01     0.45 1.01      684      788

bayesian_summary=summary(mod_thermo)
bayesian_summary_dataframe=bayesian_summary$fixed
bayesian_summary_dataframe=round(bayesian_summary_dataframe,digits = 3)
write.csv(bayesian_summary_dataframe, file = paste0("~/Desktop/University/Masters/Lab/enclosure exp/bayesian_summary_dataframe_",format(today(), "%d.%m.%y"),".csv"), row.names = T)

plot(mod_thermo)
conditional_effects(mod_thermo)


new_data<-expand.grid(true_thermo=c("sun","shade"),true_color=c("blue","red","white","unclear"),part_of_day=c("Morning","Noon","Afternoon"))


fit_thermo = fitted(mod_thermo,newdata = new_data,re_formula = NA, # ignore random effects
                    summary = TRUE, # mean and 95% CI
                    probs = c(0.05,0.95,0.25,0.75)) * 100 # convert to %
colnames(fit_thermo) = c('fit', 'se', 'lwr_90', 'upr_90','lwr_50', 'upr_50')
df_plot = cbind(new_data, fit_thermo)


tab<-table(data_thermo$true_thermo, data_thermo$true_color, data_thermo$part_of_day, data_thermo$wrong_sun.shade)
real_wrong_classifications_thermo<-as.data.frame(prop.table(tab, margin=c(1,2,3)))
real_wrong_classifications_thermo<-real_wrong_classifications_thermo[real_wrong_classifications_thermo$Var1!="unclear",]
colnames(real_wrong_classifications_thermo) <- c("true_thermo", "true_color", "part_of_day", "wrong_sun.shade", "real_wrong_classifications")
real_wrong_classifications_thermo$real_wrong_percent<-real_wrong_classifications_thermo$real_wrong_classifications*100

merged_data_thermo <- merge(df_plot, real_wrong_classifications_thermo[real_wrong_classifications_thermo$wrong_sun.shade==1,], by=c("true_thermo", "true_color", "part_of_day"))
merged_data_thermo <- subset(merged_data_thermo, select = -c(wrong_sun.shade,real_wrong_classifications))
merged_data_thermo <- merged_data_thermo[,c(1,2,3,10,4,5,6,7,8,9)]

library(plotrix)
merged_data_thermo=merged_data_thermo[merged_data_thermo$true_color!="unclear",]
plot(merged_data_thermo$fit~merged_data_thermo$real_wrong_percent)
abline(coef = c(0,1))
plotCI(merged_data_thermo$real_wrong_percent,merged_data_thermo$fit,ui=merged_data_thermo$upr_90,li=merged_data_thermo$lwr_90)

# For each part of the day plot the predicted percentages of wrong classifications of thermal location, for each color, by the bayesian model
# With the real percentages of wrong classifications generated by the manual model validation.

#Morning
morning_merged=merged_data_thermo[merged_data_thermo$part_of_day=="Morning",]

pd <- position_dodge(0.2) # move them .05 to the left and right

ggplot(morning_merged, aes(x=true_thermo,color=true_color)) + 
  geom_errorbar(aes(y=fit, ymin=lwr_90, ymax=upr_90), width=0,position=pd) +
  geom_errorbar(aes(y=fit, ymin=lwr_50, ymax=upr_50), width=0,linewidth=1,position=pd) +
  scale_color_manual(values=c( "#619CFF","#F8766D","grey")) + 
  geom_point(aes(y=real_wrong_percent,fill="Real wrong percentages"),shape=8,size=3,position=pd) +
  guides(color = guide_legend("",override.aes = list(pointtype = c(1, 1, 1), shape = c(NA,NA ,NA) ),order=1),
         fill = guide_legend("")) +
  ylab("Predicted wrong percentages") +
  xlab("Location") +
  theme_bw()


#Noon
noon_merged=merged_data_thermo[merged_data_thermo$part_of_day=="Noon",]


ggplot(noon_merged, aes(x=true_thermo,color=true_color)) + 
  geom_errorbar(aes(y=fit, ymin=lwr_90, ymax=upr_90), width=0,position=pd) +
  geom_errorbar(aes(y=fit, ymin=lwr_50, ymax=upr_50), width=0,linewidth=1,position=pd) +
  scale_color_manual(values=c( "#619CFF","#F8766D","grey")) + 
  geom_point(aes(y=real_wrong_percent,fill="Real wrong percentages"),shape=8,size=3,position=pd) +
  guides(
    color = guide_legend("",override.aes = list(pointtype = c(1, 1, 1), shape = c(NA,NA ,NA) ),order=1),
    fill = guide_legend("")) +
  ylab("Predicted wrong percentages") +
  xlab("Location") +
  theme_bw()


#Afternoon
afternoon_merged=merged_data_thermo[merged_data_thermo$part_of_day=="Afternoon",]


ggplot(afternoon_merged, aes(x=true_thermo,color=true_color)) + 
  geom_errorbar(aes(y=fit, ymin=lwr_90, ymax=upr_90), width=0,position=pd) +
  geom_errorbar(aes(y=fit, ymin=lwr_50, ymax=upr_50), width=0,linewidth=1,position=pd) +
  scale_color_manual(values=c( "#619CFF","#F8766D","grey")) + 
  geom_point(aes(y=real_wrong_percent,fill="Real wrong percentages"),shape=8,size=3,position=pd) +
  guides(
    color = guide_legend("",override.aes = list(pointtype = c(1, 1, 1), shape = c(NA,NA ,NA) ),order=1),
    fill = guide_legend("")) +
  ylab("Predicted wrong percentages") +
  xlab("Location") +
  theme_bw()




ggplot(merged_data_thermo, aes(x=true_thermo,color=true_color)) + 
  geom_errorbar(aes(y=fit, ymin=lwr_90, ymax=upr_90), width=0,position=pd) +
  geom_errorbar(aes(y=fit, ymin=lwr_50, ymax=upr_50), width=0,linewidth=1,position=pd) +
  scale_color_manual(values=c( "#619CFF","#F8766D","grey")) + 
  geom_point(aes(y=real_wrong_percent,fill="Real wrong percentages"),shape=8,size=3,position=pd) +
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 20),legend.title=element_blank(), 
        legend.text=element_text(size=18),
        axis.text=element_text(size=15), axis.title=element_text(size=18),
        axis.ticks.x=element_blank()) +
  facet_grid( ~ factor(part_of_day,levels=c('Morning','Noon','Afternoon'))) +
  theme(strip.text = element_text(size = 18),
        panel.grid = element_blank())












