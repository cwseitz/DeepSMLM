library(spatstat)
library(ggplot2)
library(gridExtra)

# Function to compute L-function and return it
computeLFunction <- function(file) {
  ############################
  # Load the dataset
  ############################
  
  # Read the CSV file
  data <- read.csv(file)
  print(nrow(data))
  #data <- data[sample(nrow(data), 1000), ] #sample N localizations
  data$xclust <- data$x_mle * 108.3
  data$yclust <- data$y_mle * 108.3
  
  ############################
  # Point pattern statistics
  ############################
  
  # Create a point pattern object
  points <- ppp(data$xclust, data$yclust, owin(range(data$xclust), range(data$yclust)))
  
  # Set the maximum distance for estimation
  r_max <- 500  # Set your desired maximum distance
  
  # Compute the L-function
  L <- Lest(points, rmax = r_max, correction = "none")
  L$un <- L$un - L$r
  
  return(L)  # Return the computed L-function
}

##################################################

dir <- '/home/cwseitz/Desktop/230914/230909'
file_list <- list.files(path = dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)

# Initialize a list to store L-functions
L_functions <- list()
combined <- data.frame(matrix(, nrow=513, ncol=0))

# Iterate over the file list, compute L-functions, and store them
for (file in file_list) {
  L <- computeLFunction(file)
  un <- L$un
  r <- L$r
  
  # Extract the base file name without extension
  col_name <- tools::file_path_sans_ext(basename(file))
  
  # Assign 'un' values to a column with the file name as the column name
  combined[[col_name]] <- un
  combined$r <- r
}

write.csv(combined, file = "combined_data.csv", row.names = FALSE)



