class A:
	def __init__(self,nameA):
		self.name = nameA
	def func0(self):
		self.name = 'func0'
		self.func1()
	def func1(self):
		print("i get self.name"+self.name)
		self.name = 'func1'
a = A('good')
print(a.name)

a.func0()
print(a.name)