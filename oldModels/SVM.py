import scanpy as sc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from dataset import *
from trainAndTest import *


microglia = sc.read_h5ad("../../data/rosmap/microglia.h5ad")
metadata = pd.read_csv("../../data/rosmap/rosmap_clinical.csv")

scores, _, annListUpdated = preprocess([microglia], metadata)
braakDict = scores[0]
inputData = annListUpdated[0]
ids = np.array(inputData.obs["individualID"].unique())

main_scores = np.array([braakDict[ind] for ind in ids])
train_ids, test_ids = train_test_split(ids, test_size=0.2, stratify=main_scores, random_state=42)


X_train = inputData[inputData.obs["individualID"].isin(train_ids)]
X_test = inputData[inputData.obs["individualID"].isin(test_ids)]
y_train = np.array([braakDict[ind] for ind in X_train.obs["individualID"]])
y_test = np.array([braakDict[ind] for ind in X_test.obs["individualID"]])

#_, index = selectKBest(X_train.X, y_train, top_k=1000)
# print(index)

X_train = X_train.X
X_test = X_test.X

# # Example: Generate dummy data
# # Replace this with your real data (e.g., embeddings from your model)
# X = np.random.randn(500, 1000)  # 500 samples, 1000 features
# y = np.random.randint(0, 4, size=500)  # 4 classes: 0, 1, 2, 3
#
#
#
# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVM model
model = xgb.XGBClassifier(
    n_estimators=100,      # number of trees
    max_depth=6,           # depth of trees
    learning_rate=0.1,     # step size shrinkage
    n_jobs=-1,             # use all CPU cores
    tree_method='hist',    # fast histogram-based training
    use_label_encoder=False,
    eval_metric='mlogloss' # good for multi-class
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))