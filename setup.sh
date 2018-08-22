#!/bin/bash
# Workspace Environment and Dataset Setup Script
# Author: Yuya J. Ong (yuyajeremyong@gmail.com)

# Dataset Setup
echo "> SETUP DATASET"
mkdir data

git clone https://github.com/laiguokun/multivariate-time-series-data.git
cd multivariate-time-series-data

gunzip electricity/electricity.txt.gz
gunzip exchange_rate/exchange_rate.txt.gz
gunzip solar-energy/solar_AL.txt.gz
gunzip traffic/traffic.txt.gz

mv electricity/electricity.txt ../data/
mv exchange_rate/exchange_rate.txt ../data/
mv solar-energy/solar_AL.txt ../data/
mv traffic/traffic.txt ../data/

cd ..
rm -rf multivariate-time-series-data

# Setup Workspace Directories
mkdir logs
mkdir chkpts
