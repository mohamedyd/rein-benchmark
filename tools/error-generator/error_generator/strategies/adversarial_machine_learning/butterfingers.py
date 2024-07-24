#clone from git https://github.com/Decagon/butter-fingers.git

from random import randint
import random

def butterfinger(text,prob=0.6,keyboard='querty'):

	keyApprox = {}
	
	if keyboard == "querty":
		keyApprox['q'] = "qwasedzx"
		keyApprox['w'] = "wqesadrfcx"
		keyApprox['e'] = "ewrsfdqazxcvgt"
		keyApprox['r'] = "retdgfwsxcvgt"
		keyApprox['t'] = "tryfhgedcvbnju"
		keyApprox['y'] = "ytugjhrfvbnji"
		keyApprox['u'] = "uyihkjtgbnmlo"
		keyApprox['i'] = "iuojlkyhnmlp"
		keyApprox['o'] = "oipklujm"
		keyApprox['p'] = "plo['ik"

		keyApprox['a'] = "aqszwxwdce"
		keyApprox['s'] = "swxadrfv"
		keyApprox['d'] = "decsfaqgbv"
		keyApprox['f'] = "fdgrvwsxyhn"
		keyApprox['g'] = "gtbfhedcyjn"
		keyApprox['h'] = "hyngjfrvkim"
		keyApprox['j'] = "jhknugtblom"
		keyApprox['k'] = "kjlinyhn"
		keyApprox['l'] = "lokmpujn"

		keyApprox['z'] = "zaxsvde"
		keyApprox['x'] = "xzcsdbvfrewq"
		keyApprox['c'] = "cxvdfzswergb"
		keyApprox['v'] = "vcfbgxdertyn"
		keyApprox['b'] = "bvnghcftyun"
		keyApprox['n'] = "nbmhjvgtuik"
		keyApprox['m'] = "mnkjloik"
		keyApprox[' '] = " "
	else:
		print ("Keyboard not supported.")

	probOfTypoArray = []
	probOfTypo = int(prob * 100)

	buttertext = ""
	for letter in text:
		lcletter = letter.lower()
		if not lcletter in keyApprox.keys():
			newletter = lcletter
		else:
			if random.choice(range(0, 100)) <= probOfTypo:
				newletter = random.choice(keyApprox[lcletter])
			else:
				newletter = lcletter
		# go back to original case
		if not lcletter == letter:
			newletter = newletter.upper()
		buttertext += newletter

	return buttertext
