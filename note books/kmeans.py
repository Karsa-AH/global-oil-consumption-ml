import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

file_path = "data\\a.csv"


data = pd.read_csv(file_path)

target_year = int(input("Enter the target year: "))

selected_data = data[data["Year"] == target_year]

features = selected_data[["Oil consumption - TWh"]]

kmeans = KMeans(n_clusters=7, random_state=42)
selected_data["Cluster"] = kmeans.fit_predict(features)

#########################################################################################################
population_file_path = "data\\population-and-demography.csv"
population_data = pd.read_csv(population_file_path)
merged_data = pd.merge(
    data, population_data, left_on=["Entity", "Year"], right_on=["Country name", "Year"]
)

merged_data["Oil consumption per capita - TWh"] = (
    merged_data["Oil consumption - TWh"] / merged_data["Population"]
)

selected_data2 = merged_data[merged_data["Year"] == target_year]

features = selected_data2[["Oil consumption per capita - TWh"]]

kmeans2 = KMeans(n_clusters=7, random_state=42)
selected_data2["Cluster"] = kmeans2.fit_predict(features)

fig, ax = plt.subplots(nrows=2, ncols=1)  # , figsize=(15, 10))
ax[0].scatter(
    selected_data["Entity"],
    selected_data["Oil consumption - TWh"],
    c=selected_data["Cluster"],
    cmap="rainbow",
)
ax[0].set_title(f"Oil Consumption Clustering for {target_year}")
ax[0].set_xlabel("Country")
ax[0].set_ylabel("Oil Consumption - TWh")
ax[1].scatter(
    selected_data2["Entity"],
    selected_data2["Oil consumption per capita - TWh"],
    c=selected_data2["Cluster"],
    cmap="rainbow",
)
ax[1].set_title(f"Oil Consumption Clustering per Capita for {target_year}")
ax[1].set_xlabel("Country")
ax[1].set_ylabel("Oil Consumption per Capita - TWh")
plt.tight_layout()
plt.show()
