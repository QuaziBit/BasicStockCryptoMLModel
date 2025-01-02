#!/bin/bash

# Enable error handling
set -e

# Define the LSTM script path
LSTM_SCRIPT_DIR="LSTM/v24_b"
LSTM_SCRIPT_NAME="lstm_ai_v24_7_daily_b.py"
LSTM_SCRIPT_NAME_NO_EXT="lstm_ai_v24_7_daily_b"
LSTM_SCRIPT_PATH="${LSTM_SCRIPT_DIR}/${LSTM_SCRIPT_NAME}"

# Define the output directories
LSTM_OUTPUT_DIR="/app/LSTM/v24_b/${LSTM_SCRIPT_NAME_NO_EXT}"
DATA_OUTPUT_DIR="/app/yahoo_finance"

# Run the main Python script based on the provided argument
if [ "$1" == "lstm" ]; then
    echo "Running LSTM script..."
    
    # Ensure the output directory exists
    if [ ! -d "${LSTM_OUTPUT_DIR}" ]; then
        echo "Creating LSTM output directory at ${LSTM_OUTPUT_DIR}"
        mkdir -p "${LSTM_OUTPUT_DIR}"
    else
        echo "LSTM output directory already exists. Retaining existing files:"
        ls -la "${LSTM_OUTPUT_DIR}"
    fi

    # Run the LSTM script
    if [ -f "${LSTM_SCRIPT_PATH}" ]; then
        echo "Running LSTM script: ${LSTM_SCRIPT_PATH}"
        python "${LSTM_SCRIPT_PATH}"
    else
        echo "Error: LSTM script not found at ${LSTM_SCRIPT_PATH}"
        exit 1
    fi

elif [ "$1" == "data" ]; then
    echo "Running data script..."
    
    # Ensure the output directory exists
    if [ ! -d "${DATA_OUTPUT_DIR}" ]; then
        echo "Creating data output directory at ${DATA_OUTPUT_DIR}"
        mkdir -p "${DATA_OUTPUT_DIR}"
    else
        echo "Data output directory already exists. Retaining existing files:"
        ls -la "${DATA_OUTPUT_DIR}"
    fi

    # Run the data script
    python yahoo_finance/get_finance_data_v7.py

else
    echo "Please specify 'lstm' or 'data'."
    exit 1
fi

# Verify the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Python script failed to execute."
    exit 1
fi

# Determine the output directory
if [ "$1" == "lstm" ]; then
    OUTPUT_DIR="${LSTM_OUTPUT_DIR}"
elif [ "$1" == "data" ]; then
    OUTPUT_DIR="${DATA_OUTPUT_DIR}"
fi

# Verify the output directory exists
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: Output directory ${OUTPUT_DIR} does not exist."
    exit 1
fi

echo "Contents of the output directory in the container:"
ls -la "${OUTPUT_DIR}"

# Copy the output to the host
if [ -d "/host/desktop" ]; then
    TARGET_DIR="/host/desktop/Docker_Output"
    echo "Desktop mount detected. Copying to: ${TARGET_DIR}"
elif [ -d "/host/root" ]; then
    TARGET_DIR="/host/root/output"
    echo "Root directory mount detected. Copying to: ${TARGET_DIR}"
else
    echo "Error: No valid host directory detected. Please map a volume when running the container."
    exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Copy the files to the target directory
echo "Copying files from ${OUTPUT_DIR} to ${TARGET_DIR}..."
cp -r "${OUTPUT_DIR}/" "${TARGET_DIR}/"

# Verify the files were copied successfully
if [ "$(ls -A ${TARGET_DIR})" ]; then
    echo "Output copied successfully to ${TARGET_DIR}"
else
    echo "Error: Output copy failed. Destination directory is empty."
    exit 1
fi

echo "All operations completed successfully!"
