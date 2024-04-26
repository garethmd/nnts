.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = rnn-convariates
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Download Data from Google Drive
sync_data_from_drive:
	$(PYTHON_INTERPRETER) -m pip install -q gdown
	gdown 1CAuj-mxzEuxE_B_sRgp4WrT1B7oRjyDh -O data/electricity_hourly_dataset.tsf
	gdown 19YT5RySccxmyyHi168i1FRCOnPM5D47w -O data/hospital_dataset.tsf
	gdown 1tcIdc580SkoeSuLITGpE-2IkE8JTw5lF -O data/tourism_monthly_dataset.tsf
	gdown 1JSMDgqB_AcoxxRlClucyXGn5BaKeSW2_ -O data/traffic_weekly_dataset.tsf
	
# Create virtual environment
covenv:
	@echo "Creating virtual environment..."
	@python3 -m venv covenv
	@echo "Virtual environment created."

# Activate virtual environment
activate:
	@echo "Activating virtual environment..."
	@. covenv/bin/activate

# Install requirements
install:
	@echo "Installing requirements..."
	@covenv/bin/pip install -r requirements.txt
	@echo "Requirements installed."

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf covenv
	@echo "Cleanup complete."

# Make all
all: covenv activate install
	@echo "Virtual environment created, activated, requirements installed"