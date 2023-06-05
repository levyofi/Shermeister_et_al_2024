# Assuming your original data frame is named `data`
library(dplyr)
library(ggplot2)
data = read.csv("model_confidence_07.05.ALL_IMAGES_29.01.23.csv", header = T)
data$time = hms(data$model_time)

#parts of the day
data$part_of_day<-ifelse(hms(data$time)<hms("11:00:00"),"Morning",ifelse(hms(data$time)>=hms("11:00:00") & hms(data$time)<hms("15:00:00"),"Noon","Afternoon"))
data$part_of_day<-as.factor(data$part_of_day)

#get enclosure, population and camera from the concatenate id
data$camera=as.factor(substr(data$Concatenate, 1, 3) )
data$enclosure = as.factor(substr(data$camera, 1, 2) )
data$population =  as.factor(substr(data$camera, 1, 1) )

#thermo model ALL DATA (NOT ONLY >95)
data_thermo = data[data$true_thermo!="unclear" & data$true_thermo!="negative" & data$true_color!="unclear",]

#thermo model - 95 conf and above
data_thermo_95 = data_thermo [ data_thermo$thermo_confidence>=95 & data_thermo$color_confidence>=95,]
nrow(data_thermo_95)/nrow(data_thermo)

#get the hour in decimal time
data_thermo_95$h = hour(data_thermo_95$time)+minute(data_thermo_95$time)/60+second(data_thermo_95$time)/3600 + runif(0, 1/3600, n = length(data_thermo_95$time)) 

#sort by camera and decimal time
data_thermo_95 = data_thermo_95[order(data_thermo_95$camera, data_thermo_95$h),]

data_thermo_95$true_thermo = as.factor(data_thermo_95$true_thermo)
data_thermo_95$true_color = as.factor(data_thermo_95$true_color)

write.csv(data_thermo_95, row.names = F, file="data_for_analysis.csv")

percentage_df <- data_thermo_95 %>%
  group_by(true_color, part_of_day, true_thermo, wrong_sun.shade) %>%
  summarize(count = n()) %>%
  mutate(total = sum(count),
         percentage = count / total * 100) %>%
  ungroup()

# Ensure part_of_day is a factor with the correct levels
percentage_df$part_of_day <- factor(percentage_df$part_of_day, levels = c("Morning", "Noon", "Afternoon"))

# Define function for custom hjust
label_hjust <- function(labels) {
  hjust_values <- ifelse(labels == "Shade", 1.5, 2.5)
  return(hjust_values)
}

plot = ggplot(percentage_df, aes(x = interaction(true_thermo, true_color), y = percentage, fill = interaction(true_color, wrong_sun.shade))) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(data = subset(percentage_df, true_thermo == "sun"), aes(label = true_color), y = 100, vjust = -0.1, hjust=1) +
  scale_fill_manual(values = c("blue.0" = "lightblue", "blue.1" = "blue",
                               "white.0" = "lightgrey", "white.1" = "grey22",
                               "red.0" = "lightcoral", "red.1" = "darkred"),
                    name = "Model Performance",
                    labels = c("Blue - Correct", "Blue - Wrong",
                               "White - Correct", "White - Wrong",
                               "Red - Correct", "Red - Wrong")) +
  facet_grid(~ part_of_day, scales = "free_x", space = "free_x") +
  labs(x = "Microhabitat", y = "Percentage of predictions (%)") +
  scale_x_discrete(labels = c("Shade", "Sun  ", "Shade", "Sun  ", "Shade", "Sun  ") ) +
  theme_minimal() +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 90, hjust = 2.25, vjust=0),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        plot.margin = margin(0, 0, -.0, 1, "cm"))  # Adjust this line

ggsave(filename = "figure 6.jpg", plot = plot, width = 8, height = 4, dpi = 300, bg = "white")

