import numpy as np

K = 4

raw = np.random.random((K,K))

target = 0.7

mult = (np.sum(raw) - np.trace(raw)) / (1/target - 1) / np.trace(raw)

for k in range(K):
    raw[k,k] = raw[k,k]*mult
    
acc = np.trace(raw) / np.sum(raw)

precision = 0
recall = 0

for k in range(K):
    precision += 1/K*(raw[k,k] / np.sum(raw[:,k]))
    recall += 1/K*(raw[k,k] / np.sum(raw[k,:]))
    
print([acc, precision, recall])