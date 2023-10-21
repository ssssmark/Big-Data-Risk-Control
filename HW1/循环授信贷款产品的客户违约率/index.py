from Solution import Solution

sln = Solution()
res = sln.compute()

data = [
    0.5904,
    0.34010840108401086,
    81550955.0,
    0.10568031402639216
]

for i in range(len(data)):
    error = res[i]-data[i]
    print("Error:"+str(error))