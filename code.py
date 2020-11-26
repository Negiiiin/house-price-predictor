#Ngin Baghbanzade
#810196599

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 


# ****************************************************************** PART A ****************************************************************


data = pd.read_csv("houses.csv")
meanOfeachColumn = data.mean()

#removing the categorical datas

toDrop = ['LotConfig', 'Neighborhood']
data.drop(toDrop, inplace = True, axis = 1)

#filling the NAN cells

toFill = ['MSSubClass', 'LotArea', 'OverallQual', 'LotFrontage', 'BedroomAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF', 'YearBuilt', 'SalePrice', 'OverallCond']
data[toFill] = data[toFill].fillna(meanOfeachColumn)

# Draw the charts

numberOfRows = len(data['Id'])	#number of all datas
colors = np.random.rand(numberOfRows)
area = 4

'''
plt.scatter(data['MSSubClass'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['LotArea'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['OverallQual'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['LotFrontage'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['OverallCond'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['BedroomAbvGr'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['TotRmsAbvGrd'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['TotalBsmtSF'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.scatter(data['YearBuilt'], data['SalePrice'], s = 4, c = colors, alpha = 0.5)
plt.show()
'''



# ****************************************************************** PART B ****************************************************************

#calcute the RMSE

X = 'TotalBsmtSF'
Y = 'SalePrice'
W = np.sum((data[X] - data.mean()[X]) * (data[Y] - data.mean()[Y]))/np.sum((data[X] - data.mean()[X])**2)
B = data.mean()[Y] - W * data.mean()[X]

estimatedPrice = np.full(numberOfRows, W * data['TotalBsmtSF'] + B)


plt.scatter(data['TotalBsmtSF'], data['SalePrice'], s = 4, c = 'blue', alpha = 0.5)
plt.scatter(data['TotalBsmtSF'], estimatedPrice, s = 4, c = 'red', alpha = 0.5)
plt.show()



theSigma = np.sum((((data['TotalBsmtSF'] * W) + B) - data['SalePrice'])**2)
#print(W)
#print(B)
RMSE = math.sqrt((1/numberOfRows) * theSigma)
#print(RMSE)



# ****************************************************************** PART E ****************************************************************

	

def KNN(input):
	standardizedValues = {'TotalBsmtSF': (data['TotalBsmtSF'] - data.min()['TotalBsmtSF']) / (data.max()['TotalBsmtSF'] - data.min()['TotalBsmtSF']), 'MSSubClass': (data['MSSubClass'] - data.min()['MSSubClass']) / (data.max()['MSSubClass'] - data.min()['MSSubClass']),
		'LotArea': (data['LotArea'] - data.min()['LotArea']) / (data.max()['LotArea'] - data.min()['LotArea']), 
		'OverallQual': (data['OverallQual'] - data.min()['OverallQual']) / (data.max()['OverallQual'] - data.min()['OverallQual']),
		'LotFrontage': (data['LotFrontage'] - data.min()['LotFrontage']) / (data.max()['LotFrontage'] - data.min()['LotFrontage']),
		'OverallCond': (data['OverallCond'] - data.min()['OverallCond']) / (data.max()['OverallCond'] - data.min()['OverallCond']),
		'BedroomAbvGr': (data['BedroomAbvGr'] - data.min()['BedroomAbvGr']) / (data.max()['BedroomAbvGr'] - data.min()['BedroomAbvGr']),
		'TotRmsAbvGrd': (data['TotRmsAbvGrd'] - data.min()['TotRmsAbvGrd']) / (data.max()['TotRmsAbvGrd'] - data.min()['TotRmsAbvGrd']),
		'YearBuilt': (data['YearBuilt'] - data.min()['YearBuilt']) / (data.max()['YearBuilt'] - data.min()['YearBuilt']),
		'SalePrice': (data['SalePrice'] - data.min()['SalePrice']) / (data.max()['SalePrice'] - data.min()['SalePrice'])}
		
		
	toStandardize = ['MSSubClass', 'LotArea', 'OverallQual', 'LotFrontage', 'BedroomAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF', 'YearBuilt', 'OverallCond']
	input[toStandardize] = (input[toStandardize] - data.min()[toStandardize]) / (data.max()[toStandardize] - data.min()[toStandardize])

	distances = np.sqrt(((standardizedValues['MSSubClass'] - input['MSSubClass'][0])**2)
	+ ((standardizedValues['LotArea'] - input['LotArea'][0])**2 + ((standardizedValues['OverallQual'] - input['OverallQual'][0])**2))
	+ ((standardizedValues['LotFrontage'] - input['LotFrontage'][0])**2) + ((standardizedValues['BedroomAbvGr'] - input['BedroomAbvGr'][0])**2)
	+ ((standardizedValues['TotRmsAbvGrd'] - input['TotRmsAbvGrd'][0])**2) + ((standardizedValues['TotalBsmtSF'] - input['TotalBsmtSF'][0])**2)
	+ ((standardizedValues['YearBuilt'] - input['YearBuilt'][0])**2) + ((standardizedValues['OverallCond'] - input['OverallCond'][0])**2))

	tenNearestIndexes = np.argpartition(distances, 10)
	average = np.sum(data['SalePrice'][tenNearestIndexes][:10])/10
	return average
	
#print(KNN())