#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the iris dataset

iris=sns.load_dataset('iris')
print(iris.head())

#dataset information
print(iris.shape)
print(iris.info())

#statistical analysis of the dataset
print(iris.describe())

#Checking for null values in the dataset
print(iris.isnull().sum())

#Checking species distribution in the dataset
print(iris['species'].unique())
print(iris['species'].value_counts())

#countplot for species distribution
sns.countplot(x='species',data=iris)
plt.show()

#scatterplot for sepal length and sepal width
sns.scatterplot(x='sepal_length',
                y='sepal_width',
                hue='species',
                data=iris
                )
plt.title('Sepal Length vs Sepal Width')
plt.show()

#boxplot for sepal length and species
sns.boxplot(x='species',
            y='sepal_length',
            data=iris
            )
plt.title('Boxplot of Sepal Length by Species')
plt.show()

#correlation heatmap
corr=iris.drop('species',axis=1)
sns.heatmap(corr,annot=True,cmap="Correlation")
plt.title('Correlation Heatmap')
plt.show()

#Violin plot for petal length and species
sns.violinplot(x='species',
               y='petal_length',
               data=iris
               )
plt.title('Violin Plot of Petal Length by Species')
plt.show()