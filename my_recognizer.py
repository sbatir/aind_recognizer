import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Initiate two lists. 
    prob, guesses = [], []
    # TODO implement the recognizer
    # return probabilities, guesses
    # X can be thought of as the sequence, L is the length. use get_all_Xlengths() to harvest sequence X and corresponding length L. 
    for X, lengths in test_set.get_all_Xlengths().values():
        llDict = {} #Initiate the word likelihood dictionary. 
        highScore, bestGuess = float('-inf'), None
        for word, model in models.items():
            try:
                score = model.score(X, lengths)
                llDict[word] = score #Store to the log likelihood dictionary of words. 
                if score > highScore:
                    highScore, bestGuess = score, word
            except: 
                llDict[word] = float('-inf')
        guesses.append(bestGuess)
        prob.append(llDict)
    return prob, guesses