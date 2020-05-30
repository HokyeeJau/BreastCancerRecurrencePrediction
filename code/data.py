import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
# from keras.utils import to_categorical

index = ['id', 'outcome', 'recur_time', 'radius_mean', 'texture_mean',
       'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
       'concavity_mean', 'concave points_mean', 'symmetry_mean',
       'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
       'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
       'concave points_se', 'symmetry_se', 'fractal_dimension_se',
       'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
       'smoothness_worst', 'compactness_worst', 'concavity_worst',
       'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
       'tumor_size', 'lymph_node_status', 'radius_distance',
       'texture_distance', 'perimeter_distance',
       'area_distance', 'smoothness_distance',
       'compactness_distance', 'concavity_distance',
       'concave points_distance', 'symmetry_distance',
       'fractal_dimension_distance']

def completeDS():
	df = np.loadtxt('../ProcessedDataset/cleaned_data.csv',dtype=str,delimiter=',')
	df = pd.DataFrame(df, columns=index)
	for i in range(2, len(index)):
		df.iloc[:, i] = df.iloc[:,i].astype(np.float32)
	return df

def completeX():
	x = np.loadtxt('../ProcessedDataset/x.csv', dtype=np.float32, delimiter=',')
	x = pd.DataFrame(x, columns=index[2:])
	x = x.astype(np.float32)
	return x

def standardX():
    x = completeX()
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x = pd.DataFrame(x, columns=index[2:])
    return x

def labeledY():
	y = np.loadtxt('../ProcessedDataset/y.csv', dtype=str, delimiter=',')
	return y

# def onehotY():
# 	y = binaryY()
# 	y = to_categorical(y)
# 	return y

def binaryY():
	y = labeledY()
	y[y=='N'] = 0
	y[y=='R'] = 1
	return y
