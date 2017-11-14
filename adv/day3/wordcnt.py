from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def wordcnt():
	positiveFiles = ['data/positiveReviews/' + f for f in listdir('data/positiveReviews/') if isfile(join('data/positiveReviews/', f))]
	negativeFiles = ['data/negativeReviews/' + f for f in listdir('data/negativeReviews/') if isfile(join('data/negativeReviews/', f))]
	numWords = []
	for pf in positiveFiles:
	    with open(pf, "r", encoding='utf-8') as f:
	        line=f.readline()
	        counter = len(line.split())
	        numWords.append(counter)       
	print('Positive files finished')

	for nf in negativeFiles:
	    with open(nf, "r", encoding='utf-8') as f:
	        line=f.readline()
	        counter = len(line.split())
	        numWords.append(counter)  
	print('Negative files finished')

	numFiles = len(numWords)
	print('The total number of files is', numFiles)
	print('The total number of words in the files is', sum(numWords))
	print('The average number of words in the files is', sum(numWords)/len(numWords))

	# plt.hist(numWords, 50)
	# plt.xlabel('Sequence Length')
	# plt.ylabel('Frequency')
	# plt.axis([0, 1200, 0, 8000])
	# plt.show()

	return numFiles

