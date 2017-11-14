
import numpy as np


# load word list
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

print('len wordsList = ', len(wordsList))
print('wordVectors.shape = ', wordVectors.shape)

baseballIndex = wordsList.index('baseball')
wordVectors[baseballIndex]
print('baseballIndex = ', baseballIndex)
print(wordVectors[baseballIndex])