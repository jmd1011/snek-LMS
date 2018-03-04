def ast(obj):
	class Snippet(object):
		def __init__(self):
			self.original = obj
			self.code = "foobar"
		def __call__(self,*args):
			return obj(*args)
	return Snippet()

def lms(obj):
	class Snippet(object):
		def __init__(self):
			self.original = obj
			self.code = "foobar"
		def __call__(self,*args):
			return obj(*args)
	return Snippet()

