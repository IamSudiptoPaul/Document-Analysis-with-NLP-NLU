#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Delete rows with missing values in the 'post' column and save the cleaned dataset to a new CSV file.

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd

df = pd.read_csv('social-media-release.csv')
df_clean = df.dropna(subset=['post'])

df_clean.to_csv('cleaned_social_media.csv', index=False)