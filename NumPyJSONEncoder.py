import json
import numpy
class NumPyJSONEncoder(json.JSONEncoder):
	def default(self,obj):
		if isinstance(obj, numpy.ndarray) and obj.ndim ==1:
			return obj.tolist()
		return json.JSONEncoder.default(self,obj)
		
		elif type(obj) == dict:
			