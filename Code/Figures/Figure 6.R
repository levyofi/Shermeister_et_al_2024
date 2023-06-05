# Load necessary libraries
library(dplyr)
library(lubridate)
library(stringr)
# Read the camera trap data
data = read.csv("model_confidence_07.05.ALL_IMAGES_29.01.23.csv", header = T)
data$time = hms(data$model_time)

#get enclosure, and camera from the concatenate id and make them a factor
data$camera=as.factor(substr(data$Concatenate, 1, 3) )
data$enclosure = as.factor(substr(data$camera, 1, 2) )

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

data <- data_thermo_95 #read.csv("data_for_analysis.csv", stringsAsFactors = T)

data$time = paste("2022-05-07", data$time) # need to add a date for the mutate to run without an error

# Convert time column to a proper datetime object
data$time <- ymd_hms(data$time)

# Sort the data by camera, true_color, and time
data <- data %>%
  arrange(camera, true_color, time)

# Add event_id using mutate and cumsum
data_events_model <- data %>%
  mutate(change_event = 
           lag(camera, default = first(camera)) != camera | lag(true_color, default = first(true_color)) != true_color | lag(thermo_label, default = first(thermo_label)) != thermo_label |
           difftime(time, lag(time, default = first(time)), units = "secs") > 10) %>%
  mutate(event_id = cumsum(change_event)) %>%
  select(-change_event)

data_events_manual <- data %>%
  mutate(change_event = 
           lag(camera, default = first(camera)) != camera | lag(true_color, default = first(true_color)) != true_color | lag(true_thermo, default = first(true_thermo)) != true_thermo |
           difftime(time, lag(time, default = first(time)), units = "secs") > 10) %>%
  mutate(event_id = cumsum(change_event)) %>%
  select(-change_event)

# Get a row for each event and wrong_sun.shade using the unique function
unique_events_model <- data_events_model %>%
  group_by(event_id) %>%
  summarise(start_time = min(time),
            end_time = max(time),
            true_color = first(true_color),
            camera = first(camera),
            thermo_label = first(thermo_label)) %>%
  ungroup()

unique_events_manual <- data_events_manual %>%
  group_by(event_id) %>%
  summarise(start_time = min(time),
            end_time = max(time),
            true_color = first(true_color),
            camera = first(camera),
            true_thermo = first(true_thermo)) %>%
  ungroup()

#plot hourly data of human labeled ("hourly_activity_zero" table) and all events predicted
# Calculate the event duration in seconds and hour of day
unique_events_model <- unique_events_model %>%
  mutate(duration = as.numeric(difftime(end_time, start_time, units = "secs")), hour = hour(start_time))

unique_events_manual <- unique_events_manual %>%
  mutate(duration = as.numeric(difftime(end_time, start_time, units = "secs")), hour = hour(start_time))

# Calculate the total duration of events for each camera, grouped by hour and condition (all rows vs. zero rows)
hourly_activity_model <- unique_events_model %>%
  group_by(camera, hour, thermo_label) %>%
  summarise(total_duration = sum(duration, na.rm = TRUE)) %>%
  mutate(condition = "model",
         shade_sun = str_to_title(thermo_label)) %>%
  ungroup()%>%
  select(-thermo_label)

hourly_activity_human <- unique_events_manual %>%
  group_by(camera, hour, true_thermo) %>%
  summarise(total_duration = sum(duration, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(condition = "human",
         shade_sun = str_to_title(true_thermo)) %>%
  select(-true_thermo)

# Print the hourly activity data frames
print(hourly_activity_model)
print(hourly_activity_human)

# Load necessary libraries
library(ggplot2)


combined_hourly_activity <- rbind(hourly_activity_human, hourly_activity_model)

# Create a new column that combines condition and shade_sun information
combined_hourly_activity <- combined_hourly_activity %>%
  mutate(condition_shade_sun = factor(paste(shade_sun, ifelse(condition == "model", "- AI", "- Obs")), levels = c("Shade - AI", "Sun - AI", "Shade - Obs", "Sun - Obs"), labels = c("Shade - AI", "Sun - AI", "Shade - Obs", "Sun - Obs")))

#a function for labels
make_labelstring <- function(mypanels) {
  mylabels <- sapply(mypanels, 
                     function(x) {paste0("(", letters[which(mypanels == x)], ")")})
  
  return(mylabels)
}

#add function to ggplot
label_panels <- ggplot2::as_labeller(make_labelstring)

# Create the faceted line chart with separate panels for each camera and separate lines for shade/sun
plot = ggplot(combined_hourly_activity, aes(x = hour, y = total_duration, color = condition_shade_sun, group = interaction(camera, condition, shade_sun))) +
  geom_line(aes(linetype = ifelse(condition == "human", "dotted", "solid")), size = 1) +
  facet_wrap(~ camera, ncol = 2, scales = "free_y", labeller = label_panels) +
  scale_color_manual(values = c("Shade - Obs" = "green", "Shade - AI" = "blue", "Sun - Obs" = "brown", "Sun - AI" = "orange"),
                     breaks = c("Shade - Obs", "Shade - AI", "Sun - Obs", "Sun - AI")) +
  guides(color = guide_legend(override.aes = list(linetype = c("dotted", "solid", "dotted", "solid"))), linetype=FALSE) +
  labs(title = NULL,
       x = "Hour of the Day",
       y = "Total Duration of Events (seconds)",
       color = "Microhabitat") +
  theme_minimal() +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = 14, face = "bold", hjust = 0.05, vjust = 0.9),  # Adjust the panel label position
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 10),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        axis.ticks = element_line(color = "black"),  # Add axis ticks
        axis.title = element_text(size = 14),  # Increase the axis titles size
        axis.text = element_text(size = 12))  # Increase the axis labels size

# Save the plot
ggsave(filename = "output_plot.png", plot = plot, dpi = 300, width = 7, height = 10, units = "in", bg = "white")


# Create the faceted line chart with separate panels for each camera and separate lines for shade/sun
plot = ggplot(combined_hourly_activity, aes(x = hour, y = total_duration, color = condition_shade_sun, group = interaction(camera, condition, shade_sun))) +
  geom_line(aes(linetype = ifelse(condition == "human", "dotted", "solid")), size = 1) +
  facet_wrap(~ camera, ncol = 2, scales = "free_y") +
  scale_color_manual(values = c("Shade - Obs" = "green", "Shade - AI" = "blue", "Sun - Obs" = "brown", "Sun - AI" = "orange"),
                     breaks = c("Shade - Obs", "Shade - AI", "Sun - Obs", "Sun - AI")) +
  guides(color = guide_legend(override.aes = list(linetype = c("dotted", "solid", "dotted", "solid"))), linetype=FALSE) +
  labs(title = NULL,
       x = "Hour of the Day",
       y = "Total Duration of Events (seconds)",
       color = "Microhabitat") +
  theme_minimal() +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = 14, face = "bold", hjust = 0.05, vjust = 0.9),  # Adjust the panel label position
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 10),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        axis.ticks = element_line(color = "black"),  # Add axis ticks
        axis.title = element_text(size = 14),  # Increase the axis titles size
        axis.text = element_text(size = 12))  # Increase the axis labels size

# Save the plot
ggsave(filename = "output_plot.png", plot = plot, dpi = 300, width = 7, height = 10, units = "in", bg = "white")
