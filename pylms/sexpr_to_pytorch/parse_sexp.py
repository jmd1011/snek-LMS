from constants import *
from utils import *

def printParsedSexp(n):
	if isinstance(n, Literal):
		print(" " + n.getValue() + " ", end = "")
	elif isinstance(n, Node):
		print("[" + n.getNodeType(), end = "")
		for el in n.getListArgs():
			printParsedSexp(el)
		print("]", end = "")
	elif isinstance(n, list):
		print(" (", end = "")
		for el in n:
			printParsedSexp(el)
		print(") ", end = "")
	else:
		print("error:")
		print(n)
		raise Exception("Malformed expression")

def parseSexp(reader):
	reader.emitDELIMS()

	# Parse literal
	if(reader.peekChar() != OPEN_NODE):
		return Literal(reader.getNextWord())

	#parse Node
	reader.acceptChar(OPEN_NODE)

	node_type = reader.getNextWord()
	list_args = []

	reader.emitDELIMS()
	while(reader.peekChar() != CLOSE_NODE):
		list_args.append(parseSexp(reader))
		reader.emitDELIMS()
	reader.acceptChar(CLOSE_NODE)

	return Node(node_type, list_args)
