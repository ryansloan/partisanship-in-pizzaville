import pandas as pd
from sklearn.decomposition import PCA

acts = ["Cheese","Pepperoni","Tofu","Jalapenos"]
legs = ["Washington","Adams","Jefferson","Madison","Monroe"]


legs_votes = pd.DataFrame.from_csv("pizzaville_votes.csv")

pca = PCA(n_components=1)
model = pca.fit(legs_votes)
scores = legs_votes.dot(model.components_[0])
print("Weights:")
print(model.components_[0])

print("Scores:")
print(scores)