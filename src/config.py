"""
config.py
---------
Central configuration module for the Boma Yetu AI Assistant.

This file is responsible for:
1. Loading configuration values from the .env file.
2. Providing safe default values where appropriate.
3. Keeping configuration in one place so other modules
   do not need to read environment variables directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# -------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------
# __file__ gives us the location of this file:
#
#     wca-ai-tool-s11-winners_04/src/config.py
#
# .parent gives us:
#
#     wca-ai-tool-s11-winners_04/src/
#
# .parent.parent gives us:
#
#     wca-ai-tool-s11-winners_04/
#
# This allows the application to find project files regardless
# of which directory the command is executed from.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------------------------
# The .env file is located in the root of the project.
#
# IMPORTANT:
# .env contains private values such as API keys.
# It must NOT be committed to Git.

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# -------------------------------------------------------------------
# APPLICATION SETTINGS
# -------------------------------------------------------------------
# Basic information about the application.

APP_NAME = os.getenv(
    "APP_NAME",
    "Boma Yetu AI Assistant"
)


# -------------------------------------------------------------------
# DIRECTORY PATHS
# -------------------------------------------------------------------
# These paths point to important project directories.
#
# Using Path instead of hard-coded strings makes the project
# easier to run on different operating systems.

DATA_DIR = PROJECT_ROOT / "data"

PDF_DIR = DATA_DIR / "pdfs"

VECTOR_DB_DIR = DATA_DIR / "vector_db"

LOG_DIR = PROJECT_ROOT / "logs"

TEST_DIR = PROJECT_ROOT / "tests"

DOCS_DIR = PROJECT_ROOT / "docs"


# -------------------------------------------------------------------
# API CONFIGURATION
# -------------------------------------------------------------------
# The actual API key will be stored in .env.
#
# We are NOT putting the real API key in this file.

API_KEY = os.getenv("API_KEY")


# -------------------------------------------------------------------
# AI MODEL CONFIGURATION
# -------------------------------------------------------------------
# We don't need to decide the final model yet.
# The model can be changed through the .env file later.

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    ""
)


# -------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------
# This function can be used by other modules to check whether
# important configuration has been loaded.

def validate_config():
    """
    Validate the basic application configuration.

    Returns:
        True if the basic configuration is valid.

    Raises:
        ValueError: If a required configuration value is missing.
    """

    if not APP_NAME:
        raise ValueError("APP_NAME is not configured.")

    return True