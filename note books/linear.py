import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("data\\a.csv")

country = input('Enter your target country: ')

selected_country_data = df[df['Entity'] == country]

X = selected_country_data[['Year']]
y = selected_country_data['Oil consumption - TWh']

model = LinearRegression()
model.fit(X, y)
plt.scatter(X, y)
plt.plot(X , model.predict(X) ,color='coral')
plt.show()


