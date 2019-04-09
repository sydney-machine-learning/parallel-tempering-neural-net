import numpy as np 
params = np.asarray([])
try:
	all_param = np.asarray(params if not ('all_param' in locals()) else np.concatenate([all_param,params],axis=0))
except ValueError:
	waste = 1
print(all_param)