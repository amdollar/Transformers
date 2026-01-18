import numpy as np

# Toy embedding (3 words, embedding_size= 4)


X = np.array([
    [1,0,0,1],
    [0,2,0,2],
    [1,1,1,1]
])

# Initilizing random weight metrics
W_Q = np.random.rand(4,4)
W_K = np.random.rand(4,4)
W_V = np.random.rand(4,4)

print(W_Q)
'''
 [0.61237623 0.28730045 0.90022056 0.9181308 ]
 [0.42063165 0.34805391 0.3439752  0.26964261]
 [0.5747164  0.06985456 0.36266617 0.34991858]]'''

# Perfor matrics multiplication using @ symbol
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print(Q)
'''
 [[1.48348817 1.11196971 0.45780471 1.20496165]
 [2.22309736 1.60593741 2.48421961 1.216907  ]
 [2.2889151  1.82863672 2.27573615 2.34214621]]'''

# How much should I pay attention to each word?
# Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) V

# Scaled dot product
dk = K.shape[1]
scores = Q @ K.T / np.sqrt(dk)

# Softmax
# It is especially important for multi-class classification problems.
# Each output value lies between 0 and 1.
# The sum of all output values equals 1.

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis= -1, keepdims= True)

attention_weight = softmax(scores)
output = attention_weight @ V

print(f'Attention Weight:  {attention_weight}')
print(f'Output: {output}')

# Attention Weight:  [[0.03311227 0.30064596 0.66624178]
#  [0.00177041 0.16645139 0.8317782 ]
#  [0.00211809 0.17576223 0.82211968]]

# Output: [[2.23857661 2.72024632 2.59280449 2.61341319]
#  [2.40491847 2.68355944 2.61215923 2.67293933]
#  [2.39642155 2.68769535 2.61285533 2.67059946]]