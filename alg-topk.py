import numpy as np

array = np.random.randint(1, 10, 10)

indexs = np.argsort(-array)

orders = indexs.argsort()

mask = orders < 3

print("asdf")