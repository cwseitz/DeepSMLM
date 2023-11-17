# Load necessary libraries
library(ggplot2)
library(dplyr)
library(stringr)

# Set the path to your files
path_to_files <- "/home/cwseitz/Desktop"

# List of file names
#file_names <- c(
#  "230816_Hela_JQ1_25uM_2h_counts",
#  "230816_Hela_JQ1_25uM_6h_counts",
#  "230816_Hela_JQ1_25uM_9h_counts",
#  "230816_Hela_JQ1_40uM_2h_counts",
#  "230816_Hela_JQ1_40uM_6h_counts",
#  "230816_Hela_JQ1_40uM_9h_counts",
#  "230816_Hela_JQ1_40uM_0h_counts"
#)

file_names <- c(
  "230816_Hela_JQ1_25uM_9h_counts",
  "230816_Hela_JQ1_40uM_9h_counts",
  "230816_Hela_JQ1_40uM_0h_counts"
)

# Read files and store data in a list
data_list <- lapply(file_names, function(file) {
  file_path <- file.path(path_to_files, paste0(file, ".csv"))
  data <- read.csv(file_path)
  data$file_name <- file  # Add the file name as a column
  return(data)
})

# Combine data from different concentrations and time points
combined_data <- do.call(rbind, data_list)

# Extract concentration and time information from the file names
combined_data$Concentration <- str_extract(combined_data$file_name, "25uM|40uM")
combined_data$Time <- str_extract(combined_data$file_name, "\\d+h")

# Calculate means and standard errors


summary_data <- combined_data %>%
  filter(Time != "0h") %>%
  group_by(Concentration, Time) %>%
  summarise(mean_counts = mean(counts),
            se_counts = sd(counts) / sqrt(n()))

print(summary_data)
# Calculate the inhibition percentage using the control counts from "40uM 0h"
control_counts <- 107

summary_data <- summary_data %>%
  mutate(inhibition_percent = 100*mean_counts / control_counts)

# Plotting

p <- ggplot(summary_data, aes(x = Time, y = inhibition_percent, fill = Concentration)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.35), color = "black", width = 0.25) +
  geom_errorbar(aes(ymin = ifelse(inhibition_percent > 0, inhibition_percent, 0),
                    ymax = inhibition_percent + 100*se_counts/control_counts),
                position = position_dodge(width = 0.35), width = 0.25, color = "black") +
  theme_minimal() +
  labs(title = "",
       x = element_blank(),
       y = "Relative BRD4 puncta (%)",
       fill = "Concentration") +
  scale_fill_manual(values = c("25uM" = "black", "40uM" = "gray")) +
  theme(panel.grid = element_blank(),          # Remove background grid
        panel.border = element_blank(),        # Remove panel border
        axis.line = element_line(color = "black"),  # Black axis lines
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", family = "Arial"),  # Slanted and bold x-axis labels
        axis.text.y = element_text(face = "bold", family = "Arial"),  # Bold y-axis labels
        legend.position = "top",               # Move legend to top
        legend.title = element_blank(),       # Remove legend title
        plot.margin = margin(1, 1, 1, 1, "cm")) # Adjust plot margins

# Print the plot
print(p)
