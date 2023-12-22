def altTabArrangement(N, K, applications):
    # Find the index of the most recently used application
    m  = applications[K-1]
    ans = []
    ans.append(m)
    for j in range(N):
        if j
    

# Input
N = 4
K = 3
applications = [1, 2, 3, 4]

# Calculate and print the result
result = altTabArrangement(N, K, applications)
print(result)
