"""
This file is used to store the settings for the app.
"""

import os

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.abspath(os.path.join(root_dir, os.pardir)), "data")

secret_path = os.path.join(os.path.abspath(os.path.join(root_dir, os.pardir)), "secret.yaml")
env_path = os.path.join(os.path.abspath(os.path.join(root_dir, os.pardir)), ".env")
