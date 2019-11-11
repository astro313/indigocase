# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Your objective is to use any method you want to develop an automatic pipeline with the training data (“train.tif” and “train_ground_truth.shp”) and to count the number of plants in the “test.tif” file as accurately as possible. 
#
# However, building a highly accurate model is only part of the story. We also want to know how you do it. As such, we’ll also be looking for you to deliver a 5-page slides on your approach. Additionally, we’d like you to put your codes in a zip file and send to us which should to include python files (.py) that can be run in command line and a notebook file (.ipynb) to walk through the pipeline. 
#
# The final component of the evaluation is to generate georeferenced maps using the stand count results from your automatic pipeline: a plant population map (the unit for the pixel is plants/ac) and a plant size map (the unit for the pixel is cm).  

import json
import os
import geojson
import matplotlib.pyplot as plt
import numpy as np
imp


