def f1(x):
	print(x)

def f2(x):
	print(x*2)

class Test:
	def __init__(self, act=f1):
		self.act = act

	def set_act(self, function):
		self.act = function

	def activate(x):
		self.act(x)

t = Test(f2)
t.act(2)
t.set_act(f1)
t.act(2)