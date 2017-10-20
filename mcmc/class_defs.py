class parameter:
	def __init__(self, name, pars):
		if len(pars) == 3:
			self.param_name = name
			self.param_val = pars[0]
			self.param_mean = pars[1]
			self.param_std_dev = pars[2]
		else:
			self.param_name = name
			self.param_val = 0
			self.param_mean = 0
			self.param_std_dev = 0

	def set_pars(self, pars):
		self.param_val = pars[0]
		self.param_mean = pars[1]
		self.param_std_dev = pars[2]

	def set_name(self, name):
		self.param_name = name

	def get_name(self):
		return self.param_name

	def get_val(self):
		return self.param_val

	def get_mean(self):
		return self.param_mean

	def get_std_dev(self):
		return self.param_std_dev

