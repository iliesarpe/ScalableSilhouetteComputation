import pandas as pd
import numpy as np
import sys
import csv
from itertools import count

def remapStocks(stock, Map, Id):
	if stock in Map:
		return Map[stock]
	else:
		ID = next(counter)
		Map[stock] = ID
		return Map[stock]


filename = sys.argv[1]
if(filename == "loc-gowalla_totalCheckins.txt" or filename=="gowTest.txt"):
	df = pd.read_csv(filename, sep=r'[\t;,]', engine='python', header=None, usecols=[2,3])
	fileout = filename.split(".")[0] + "_preprocessed.csv"
	df.to_csv(fileout, sep=',', index=False, header=False, encoding='utf-8')
	print(df.columns)
if(filename == "bio_train.dat"):
	df = pd.read_csv(filename, sep=r'[\t ]+', engine='python', header=None)
	df1 = df.drop([0, 1, 2], axis=1)
	fileout = "bioKDD2004_preprocessed.csv"
	df1.to_csv(fileout, sep=',', index=False, header=False, encoding='utf-8')
	print(df1.columns)
if(filename == "12859_2005_912_MOESM6_ESM.csv"):
	df = pd.read_csv(filename, sep=' ', engine='python', header=None)
	df1 = df.drop([0], axis=1)
	fileout = "RNA_preprocessed.csv"
	df1.to_csv(fileout, sep=',', index=False, header=False, encoding='utf-8')
if(filename == "shuttle"):
	df = pd.read_csv("shuttle.trn", sep=' ', engine='python', header=None)
	df1 = pd.read_csv("shuttle.tst", sep=' ', engine='python', header=None)
	df2 = df.drop([9], axis=1)
	df3 = df1.drop([9], axis=1)
	dfFin = pd.concat([df2,df3],ignore_index=True)
	fileout = "shuttle_preprocessed.csv"
	dfFin.to_csv(fileout, sep=',', index=False, header=False, encoding='utf-8')
if(filename == "breast"):
	df = pd.read_csv("wdbc.data", sep=",", header=0)
	df2 = df.drop(df.columns[:2], axis=1)
	print(df2)
	df2.to_csv("real/breast.csv", sep=',', index=False, header=False, encoding='utf-8')
if(filename == "wine"):
	df = pd.read_csv("winequality-white.csv", sep=";", header=0)
	df1 = pd.read_csv("winequality-red.csv", sep=";", header=0)
	full_wine = pd.concat([df, df1])
	df2 = full_wine.drop(full_wine.columns[-1], axis=1)
	print(df2)
	df2.to_csv("real/wine.csv", sep=',', index=False, header=False, encoding='utf-8')
if(filename == "powerhouse"):
	df = pd.read_csv("household_power_consumption.txt", sep=";", header=0, low_memory=False)
	df = df.drop(df.columns[[0, 1]], axis=1)
	# Replace known placeholders for missing data
	df.replace(['?', 'NA', ''], np.nan, inplace=True)

	# Convert columns 2 to 7 (based on zero indexing) to numeric, coercing errors to NaN
	cols_to_convert = df.columns[2:8]
	df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
	df.fillna(0, inplace=True)  # Replace all NaNs with 0
	df.to_csv("real/powerconsumption.csv", sep=',', index=False, header=False, encoding='utf-8')
if(filename == "metro"):
	df = pd.read_csv("Metro.csv", sep=",", header=0, low_memory=False)
	df = df.drop(df.columns[[0, 1]], axis=1)
	df = df.iloc[:, :-8]
	df.to_csv("real/metro.csv", sep=',', index=False, header=False, encoding='utf-8')
if(filename == "defaultcredit"):
	df = pd.read_csv("default.csv", sep=',', header=None, low_memory=False)
	df = df.drop(df.columns[[0,-1]], axis=1)
	df.to_csv("real/credit.csv", sep=',', index=False, header=False, encoding='utf-8')
if(filename == "iot"):
	df = pd.read_csv("RT_IOT2022", sep=',', header=0, low_memory=False)
	df = df.drop(df.columns[[0,1,2,3,4,-1,-2]], axis=1)
	df.to_csv("real/IOT.csv", sep=',', index=False, header=False, encoding='utf-8')
