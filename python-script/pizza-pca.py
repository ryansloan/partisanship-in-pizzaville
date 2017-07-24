import pandas as pd
from sklearn.decomposition import PCA

#read in legislator and vote matrix
legs_votes = pd.DataFrame.from_csv("pizzaville_votes.csv")

pca = PCA(n_components=1)
model = pca.fit(legs_votes)

#The component eigenvectors are in model.components_. 
#Take the first and dot it with each legislator's votes
scores = legs_votes.dot(model.components_[0])


print("Weights:")
print(model.components_[0])

print("Scores:")
print(scores)