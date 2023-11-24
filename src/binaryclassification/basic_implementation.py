import os

cwd = os.path.dirname(__file__)

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# -- hyperparameters
vocab_size = 10000
embedding_dim = 100
num_heads = 4
hidden_dim = 64
learning_rate = 0.001
epochs = 75


# -- create dummy data
batch_size = 1024
sequence_length = 20
x = torch.randint(
    0, vocab_size, (batch_size, sequence_length)
)  # shape (batch_size, sequence_lenght)
y = torch.randint(0, 2, (batch_size, 1)).float()  # binary labels


# -- model
class MyBinaryClassifier(nn.Module):
    """
    An multi-head attention layer on embeddings learned from input vectors, followed by a logistic classifier
    """

    def __init__(self, embedding_dim, vocab_size, hidden_dim, num_heads) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )  # lookup table shape (vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )  # view each embedding as a sequence and apply self attention
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # -- recieves x.shape = (batch_size, sequence_length)
        # -- embedding of input strings; generates (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)
        # -- attention; generates (batch_size, sequence_length, embedding_dim)
        attn_output, _ = self.attention(x, x, x)
        # -- avg pooling on sequence length dimension; generates (batch_size, embedding_dim)
        x = attn_output.mean(dim=1)
        # -- logistic classifier top; generates (batch_size, 1) with probabilities in [0, 1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


model = MyBinaryClassifier(
    embedding_dim=embedding_dim,
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

"""
Training
"""
losses = []
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch = {epoch+1} : Loss = {loss}")
    losses.append(loss.detach().numpy().item())


save_path = os.path.join(cwd, "results/binary_classification.png")
plt.figure(figsize=(7, 7))
plt.plot(range(epochs), losses)
plt.xlabel("epochs", fontsize=20)
plt.ylabel("BCE Loss", fontsize=20)
plt.tight_layout()
plt.savefig(save_path)


"""
Evaluation
"""


def binarize(prediction, threshold=0.5):
    if prediction <= threshold:
        return 0
    else:
        return 1


def evaluate_model(threshold, verbose=False):
    with torch.no_grad():
        model.eval()
        predictions = model(x)

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(batch_size):
            pi = binarize(predictions[i].item(), threshold=threshold)
            ti = y[i].int().item()

            if verbose:
                print(f"{pi} ? {ti}")
            if pi == 1 and ti == 1:
                TP += 1
            elif pi == 1 and ti == 0:
                FP += 1
            elif pi == 0 and ti == 1:
                FN += 1
            elif pi == 0 and ti == 0:
                TN += 1

    # -- confusion matrix
    confusion_matrix = defaultdict(int)
    confusion_matrix["TP"] = TP
    confusion_matrix["FP"] = FP
    confusion_matrix["TN"] = TN
    confusion_matrix["FN"] = FN
    # -- metrics
    metrics = defaultdict(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall > 0)
        else 0.0
    )
    TPR = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # True Positive Rate
    FPR = FP / (TN + FP) if (TN + FP) > 0 else 0.0  # False Positive Rate
    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1-score"] = f1
    metrics["TPR"] = TPR
    metrics["FPR"] = FPR

    return confusion_matrix, metrics


threshold = 0.5
confusion_matrix, metrics = evaluate_model(threshold=threshold)
print(*confusion_matrix.items(), sep="\t")
print(*metrics.items(), sep="\t")
print("done!")

# -- compute ROC
thresholds = np.linspace(0.2, 0.8, 10)
TPRs = []
FPRs = []
for threshold in thresholds:
    confusion_matrix, metrics = evaluate_model(threshold=threshold)
    TPRs.append(metrics["TPR"])
    FPRs.append(metrics["FPR"])

save_path = os.path.join(cwd, "results/ROC.png")
plt.figure(figsize=(7, 7))
plt.scatter(FPRs, TPRs)
plt.xlabel("FPR", fontsize=20)
plt.ylabel("TPR", fontsize=20)
plt.tight_layout()
plt.savefig(save_path)
