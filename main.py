import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC

dat = pd.read_csv('data.csv')
print(dat)

# Specify inputs for the model
ingredients = dat[['Calories', 'Sugar']].values
type_label = np.where(dat['Type'] == 'Fruit', 0, 1)

# Feature names
food_features = dat.columns.values[1:].tolist()
food_features

model = svm.SVC(kernel='linear', gamma="scale")
model.fit(ingredients, type_label)




def fruits_or_veggies(calories, sugar):
    if (model.predict([[calories, sugar]])) == 0:
        print('This is a Fruit!')
    else:
        print('This is a Vegetable!')


print();
print("Corn: ")
fruits_or_veggies(99, 1)
print("Pineapple: ")
fruits_or_veggies(83.6, 16.4)
print("Lime: ")
fruits_or_veggies(89, 11)

model2 = svm.SVC(kernel='rbf', gamma="scale")
model2.fit(ingredients, type_label)


def fruits_or_veggies2(calories, sugar):
    if (model2.predict([[calories, sugar]])) == 0:
        print('This is a Fruit!')
    else:
        print('This is a Vegetable!')


print();
print("Corn: ")
fruits_or_veggies2(99, 1)
print("Pineapple: ")
fruits_or_veggies2(83.6, 16.4)
print("Lime: ")
fruits_or_veggies2(89, 11)
