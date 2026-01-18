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

