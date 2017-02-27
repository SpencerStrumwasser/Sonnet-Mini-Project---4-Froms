from nltk.corpus import cmudict
from fix_words import load_dict_from_file
from fix_words import update_vowels

d = cmudict.dict()
#create the dictionary of words not in cmudict
dic = load_dict_from_file("dict_supp.txt")
update_vowels(dic)

def nsyl(word):
	lis = []
	try:
		#try and load the word if in cmudic
		thing = d[word]
		for y in thing:
			sums = 0
			#each stress/vowel adds one more to the syllable count
			for x in y:
					if '0' == x[-1]:
						sums += 1		
					elif '1' == x[-1]:
						sums += 1
					elif '2' == x[-1]:
						sums += 1
			lis.append(sums)
	except KeyError:
		#if not in cmu dict use dictionary
		thing = dic[word]
		sums = 0
		#each stress/vowel adds one more to the syllable count
		for x in thing:
				if '0' == x[-1]:
					sums += 1		
				elif '1' == x[-1]:
					sums += 1
				elif '2' == x[-1]:
					sums += 1
		lis.append(sums)
	return lis 

def list_meter(word):
	lis = []
	try:
		#check if word is part of nltk / cmudict
		for y in d[word]:
			#create the string of stresses in the word using 1 for stress
			#and 0 for no stress
			string = ''
			for x in y:
				if '0' == x[-1]:
					string = string + '0'
				elif '1' == x[-1]:
					string = string + '1'
				elif '2' == x[-1]:
					string = string + '0'
			lis.append(string)
	except KeyError:
		#if not in cmu dict use dictionary
		thing = dic[word]
		string = ''
		for x in thing:
			#create the string of stresses in the word using 1 for stress
			#and 0 for no stress
				if '0' == x[-1]:
					string = string + '0'		
				elif '1' == x[-1]:
					string = string + '1'
				elif '2' == x[-1]:
					string = string + '0'
		lis.append(string)
	return lis
