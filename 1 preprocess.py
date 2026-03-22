#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Finding missing values and understanding the dataset structure.

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd

# Load the dataset to begin our analysis
df = pd.read_csv('social-media-release.csv')

# Check basic info and look for any missing values
print(df.info(),'\n')

# Count how many posts are labeled real (True) vs fake (False)
print(df['class_label'].value_counts(), '\n')

# Count unique headlines to verify the dataset description
print("Unique headlines:", df['news_headline'].nunique(), '\n')

# Identify any empty fields in the posts
print("Missing values:\n")
print(df.isnull().sum())