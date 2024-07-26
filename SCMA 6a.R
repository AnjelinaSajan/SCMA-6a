# Load necessary libraries
library(tidyverse)
library(lubridate)
library(imputeTS)
library(forecast)
library(ggplot2)



# Load the dataset
nvda_data <- read.csv("C:\\Users\\dhanv\\Downloads\\NVDA Historical Data (2).csv")

# Convert the Date column to Date type using parse_date_time function
nvda_data$Date <- parse_date_time(nvda_data$Date, orders = c("mdy", "dmy", "ymd"))

# Check for missing dates and remove rows with missing dates
nvda_data <- nvda_data %>% drop_na(Date)


# Convert Vol. and Change % from string to numeric
nvda_data$Vol. <- as.numeric(gsub("M", "", nvda_data$Vol.)) * 1e6
nvda_data$Change. <- as.numeric(gsub("%", "", nvda_data$Change.))



# Check for missing values in other columns
sum(is.na(nvda_data))


# Interpolate missing values if there are any
nvda_data <- na_interpolation(nvda_data)


# Check for outliers using boxplot
boxplot(nvda_data$Price, main="Boxplot for Price", ylab="Price")


# Plot the data
ggplot(nvda_data, aes(x = Date, y = Price)) +
  geom_line() +
  labs(title = "NVIDIA Stock Price Over Time", x = "Date", y = "Price")



# Split the data into training and testing sets
split_date <- as.Date("2023-12-31")
train_data <- nvda_data %>% filter(Date <= split_date)
test_data <- nvda_data %>% filter(Date > split_date)


# Convert the data to monthly
nvda_data_monthly <- nvda_data %>%
  group_by(month = floor_date(Date, "month")) %>%
  summarise(Price = mean(Price))


# Create time series object
nvda_ts <- ts(nvda_data_monthly$Price, start = c(2020, 1), frequency = 12)


# Decompose the time series using additive model
decomp_additive <- decompose(nvda_ts, type = "additive")


# Decompose the time series using multiplicative model
decomp_multiplicative <- decompose(nvda_ts, type = "multiplicative")



# Plot the decomposed components for additive model
autoplot(decomp_additive) +
  ggtitle("Additive Decomposition of NVIDIA Stock Price") +
  theme_minimal()


# Plot the decomposed components for multiplicative model
autoplot(decomp_multiplicative) +
  ggtitle("Multiplicative Decomposition of NVIDIA Stock Price") +
  theme_minimal()


# Print a message to indicate completion
print("Data cleaning, interpolation, plotting, and decomposition are complete.")




## UNIVARIATE ANALYSIS

# Create time series objects
nvda_ts_daily <- ts(nvda_data$Price, start = c(2020, 1), frequency = 365.25)
nvda_ts_monthly <- ts(nvda_data_monthly$Price, start = c(2020, 1), frequency = 12)


# 1. Holt-Winters model and forecast for the next year
hw_model <- HoltWinters(nvda_ts_monthly)
hw_forecast <- forecast(hw_model, h = 12)
autoplot(hw_forecast) +
  ggtitle("Holt-Winters Forecast for NVIDIA Stock Price") +
  theme_minimal()



# 2. Fit ARIMA model to the daily data
arima_model_daily <- auto.arima(nvda_ts_daily)
summary(arima_model_daily)


# Diagnostic check for ARIMA model
checkresiduals(arima_model_daily)


# Fit SARIMA model to the daily data
sarima_model_daily <- auto.arima(nvda_ts_daily, seasonal = TRUE)
summary(sarima_model_daily)


# Compare ARIMA and SARIMA models
arima_aic <- AIC(arima_model_daily)
sarima_aic <- AIC(sarima_model_daily)
print(paste("ARIMA AIC:", arima_aic))
print(paste("SARIMA AIC:", sarima_aic))


# Select the best model based on AIC
if (arima_aic < sarima_aic) {
  best_model_daily <- arima_model_daily
} else {
  best_model_daily <- sarima_model_daily
}


# Ensure the best model is a valid forecast model
if (!inherits(best_model_daily, "Arima")) {
  stop("The selected best model is not a valid ARIMA model")
}

# Forecast for the next 90 days
daily_forecast <- forecast(best_model_daily, h = 90)



# Check if forecast object is created correctly
if (!inherits(daily_forecast, "forecast")) {
  stop("Forecast object was not created correctly")
}


# Plot the forecast
autoplot(daily_forecast) +
  ggtitle("Daily Forecast for NVIDIA Stock Price") +
  theme_minimal()



# 3. Fit ARIMA model to the monthly series
arima_model_monthly <- auto.arima(nvda_ts_monthly)
summary(arima_model_monthly)


# Forecast the monthly series
monthly_forecast <- forecast(arima_model_monthly, h = 12)
autoplot(monthly_forecast) +
  ggtitle("Monthly ARIMA Forecast for NVIDIA Stock Price") +
  theme_minimal()



## MULTIVARIATE


# Install necessary packages if not already installed
if (!require(tensorflow)) install.packages("tensorflow")
if (!require(caret)) install.packages("caret")
if (!require(dplyr)) install.packages("dplyr")
if (!require(purrr)) install.packages("purrr")
if (!require(keras)) install.packages("keras")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(rpart)) install.packages("rpart")
if (!require(randomForest)) install.packages("randomForest")

# Load necessary libraries
library(tensorflow)
library(caret)
library(dplyr)
library(purrr)
library(keras)
library(ggplot2)
library(rpart)
library(randomForest)

# Function to convert volume strings to numeric values
convert_volume <- function(volume_str) {
  volume_str <- toupper(gsub(",", "", volume_str))  # Remove commas and convert to uppercase
  if (grepl("B", volume_str)) {
    return(as.numeric(gsub("B", "", volume_str)) * 1e9)
  } else if (grepl("M", volume_str)) {
    return(as.numeric(gsub("M", "", volume_str)) * 1e6)
  } else if (grepl("K", volume_str)) {
    return(as.numeric(gsub("K", "", volume_str)) * 1e3)
  } else {
    return(as.numeric(volume_str))  # Handle numbers without suffix
  }
}

# Load the dataset
nvda_data <- read.csv("C:\\Users\\dhanv\\Downloads\\NVDA Historical Data (2).csv")

# Apply the function to the 'Vol.' column
nvda_data$Vol. <- sapply(nvda_data$Vol., convert_volume)

# Convert 'Change..' column to numeric
nvda_data$Change.. <- as.numeric(gsub("%", "", trimws(nvda_data$Change..)))

# Preparing data for LSTM
scaler <- preProcess(nvda_data[, "Price", drop = FALSE], method = c("range"))
scaled_data <- predict(scaler, nvda_data[, "Price", drop = FALSE])

# Split the data into train and test sets
train_size <- floor(nrow(scaled_data) * 0.8)
train_data <- scaled_data[1:train_size, , drop = FALSE]
test_data <- scaled_data[(train_size + 1):nrow(scaled_data), , drop = FALSE]

# Create sequences of 60 days data for LSTM
create_sequences <- function(data, seq_length) {
  X <- list()
  y <- numeric()
  for (i in 1:(nrow(data) - seq_length)) {
    X[[i]] <- data[i:(i + seq_length - 1), , drop = FALSE]
    y[i] <- data[i + seq_length, ]
  }
  return(list(X = array(unlist(X), dim = c(length(X), seq_length, 1)), y = y))
}

create_sequences <- function(data, seq_length) {
  n <- nrow(data)
  
  if (n <= seq_length) {
    stop("Not enough data to create sequences. Increase the length of your dataset.")
  }
  
  X <- list()
  y <- numeric()
  
  for (i in 1:(n - seq_length)) {
    if ((i + seq_length - 1) <= n) {
      X[[i]] <- data[i:(i + seq_length - 1), , drop = FALSE]
      y[i] <- data[i + seq_length, ]
    }
  }
  
  if (length(X) > 0) {
    # Handle single-column case
    if (ncol(data) == 1) {
      return(list(X = array(unlist(X), dim = c(length(X), seq_length, 1)), y = y))
    } else {
      return(list(X = array(unlist(X), dim = c(length(X), seq_length, ncol(data))), y = y))
    }
  } else {
    stop("No sequences created. Check the length of your data and sequence length.")
  }
}

# Test with train_data
train_sequences <- create_sequences(train_data, seq_length)

# Test with test_data
test_sequences <- create_sequences(test_data, seq_length)
# Reduce sequence length
nrow(test_data)

# Define a smaller sequence length suitable for your data
seq_length <- 5  # Adjust based on your dataset

# Create sequences for train data
train_sequences <- create_sequences(train_data, seq_length)

# Create sequences for test data
test_sequences <- create_sequences(test_data, seq_length)

# Check the results
str(train_sequences)
str(test_sequences)

# Define a smaller sequence length suitable for your data
seq_length <- 5  # Adjust based on your dataset

# Create sequences for train and test data
train_sequences <- create_sequences(train_data, seq_length)
test_sequences <- create_sequences(test_data, seq_length)

# Build LSTM model
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(seq_length, 1)) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dense(units = 1)

# Prepare example data (adjust this to your actual data)
X_train <- array(train_sequences$X, dim = c(length(train_sequences$X), seq_length, 1))
y_train <- train_sequences$y
X_test <- array(test_sequences$X, dim = c(length(test_sequences$X), seq_length, 1))
y_test <- test_sequences$y

# Compile the model
lstm_model %>% compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model
lstm_model %>% fit(X_train, y_train, epochs = 10, batch_size = 1)  # Adjust batch_size if needed

# Forecast using LSTM
lstm_predictions <- lstm_model %>% predict(X_test)

# Plot LSTM predictions
plot(nvda_data$Date[(nrow(nvda_data) - length(y_test) + 1):nrow(nvda_data)], 
     y_test, type='l', col='blue', 
     xlab='Date', ylab='Price', 
     main='LSTM Forecast for NVIDIA Stock Price')
lines(nvda_data$Date[(nrow(nvda_data) - length(y_test) + 1):nrow(nvda_data)], 
      lstm_predictions, col='red')
legend("topright", legend=c("True Price", "LSTM Predictions"), col=c("blue", "red"), lty=1)

# Preparing data for tree-based models
X <- nvda_data[, !names(nvda_data) %in% c("Price")]
y <- nvda_data$Price

# Split the data into train and test sets
set.seed(42)
train_index <- createDataPartition(y, p=0.8, list=FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Fit Decision Tree model
dt_model <- rpart(y_train ~ ., data = data.frame(y_train, X_train))
dt_predictions <- predict(dt_model, X_test)

# Fit Random Forest model
rf_model <- randomForest(x = X_train, y = y_train, ntree=100)
rf_predictions <- predict(rf_model, X_test)

# Evaluate models
dt_mse <- mean((y_test - dt_predictions)^2)
rf_mse <- mean((y_test - rf_predictions)^2)
cat(sprintf("Decision Tree MSE: %f\n", dt_mse))
cat(sprintf("Random Forest MSE: %f\n", rf_mse))

# Plot Decision Tree predictions
plot(nvda_data$Date[(nrow(nvda_data) - length(y_test) + 1):nrow(nvda_data)], 
     y_test, type='l', col='blue', 
     xlab='Date', ylab='Price', 
     main='Decision Tree Forecast for NVIDIA Stock Price')
lines(nvda_data$Date[(nrow(nvda_data) - length(y_test) + 1):nrow(nvda_data)], 
      dt_predictions, col='red')
legend("topright", legend=c("True Price", "Decision Tree Predictions"), col=c("blue", "red"), lty=1)

# Plot Random Forest predictions
plot(nvda_data$Date[(nrow(nvda_data) - length(y_test) + 1):nrow(nvda_data)], 
     y_test, type='l', col='blue', 
     xlab='Date', ylab='Price', 
     main='Random Forest Forecast for NVIDIA Stock Price')
lines(nvda_data$Date[(nrow(nvda_data) - length(y_test) + 1):nrow(nvda_data)], 
      rf_predictions, col='red')
legend("topright", legend=c("True Price", "Random Forest Predictions"), col=c("blue", "red"), lty=1)
