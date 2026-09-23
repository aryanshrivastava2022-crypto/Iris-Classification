# Iris-Classification 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Iris.csv')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score, adjusted_rand_score

sns.set_theme(style="whitegrid")

X = df.drop(columns=['Id', 'Species'])
y = df['Species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

print("Supervised Learning Evaluation")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, clusters)

print("\n Unsupervised Learning Evaluation")
print(f"Silhouette Score: {sil_score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

### Supervised - Confusion
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", ax=axes[0])
axes[0].set_title("Supervised: Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

### Unsupervised - K-Means 
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis", s=50)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
axes[1].set_title("Unsupervised: K-Means Clusters")
axes[1].legend()

plt.tight_layout()
plt.savefig('iris_classification_vs_clustering.png', dpi=300, bbox_inches='tight')
plt.show() 

# Output
<img width="1366" height="768" alt="Output" src="https://github.com/user-attachments/assets/c62681f8-dc43-403a-8e94-a04b1f3146e3" />
