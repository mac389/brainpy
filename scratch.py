import nltk
#from nltk.book import *

data = 'Dr. Nick Genes is a doctor.'
tokenized = data.split()
answer = [''.join([letter for letter in word if letter.isalpha()]) for word in tokenized]
print answer