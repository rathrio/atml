import time 
import torch
# run 100 iterations
for ind in range(100):
	# delay for one second
	time.sleep(0.2)
	# create a 2x2 random tensor
	x = torch.randn(2, 2)
	# print out the tensor
	print(x)
