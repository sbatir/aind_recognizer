import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bestModel, lowestBIC = None, float('inf') #Init model and BIC values. 
        
        #Iterate through all comments. 
        for i in range(self.min_n_components, self.max_n_components + 1): 
            try: #Perform try/except for each ith component. 
                model= self.base_model(i) 
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                p = i**2 + 2*i*model.n_features - 1  #Multiply out the ith component across. 
                bic = -2 *logL + p*logN 
                if bic < lowestBIC: #Remember BIC is reverse, want smallest BIC score. 
                    bestModel, lowestBIC = model, bic 
            except:
                continue
        return bestModel if bestModel else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #Similar structure to BIC. 
        bestModel, highestDIC = None, float('-inf')
        #Like BIC, iterate through every i component. 
        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(i)
                scoreList = []
                for word, (X, lengths) in self.hwords.items(): 
                    #Access dictionary hwords.items()
                    if word != self.this_word:
                        #If word not in existing dictionary for model, then add to scoreList for later removal from likelihood computation. 
                        scoreList.append(model.score(X, lengths))
                #Remove mean of collective scores of the model's subsequent sequence and length. 
                #Take current HMM base model, remove mean of log likelihoods for past models (second DIC term)
                dic = model.score(self.X, self.lengths) - np.mean(scoreList)
                #Intuition captured by line 124 from Biem et al., "to select the model that is the less likely to have generated data belonging to competing classification categories
                if dic > highestDIC:
                    bestModel, highestDIC = model, dic
            except:
                continue
        return bestModel if bestModel else self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        bestModel, bestCV = None, float("-inf")
        
        for i in range(self.min_n_components, self.max_n_components +1):
            try:
                scoreList = []
                model = self.base_model(i)
                # Examined KFold documentation to specify number of splits. 
                splitMethod = KFold(n_splits=n_splits)
                for train_i, test_i in splitMethod.split(self.sequences):
                    self.X, self.lengths = combine_sequences(train_i, self.sequences)
                    trainModel = self.base_model(i) #Train a model for each ith component. 
                    X, lengths = combine_sequences(test_i, self.sequences)
                    scoreList.append(trainModel.score(X, lengths))
                crossVal = np.mean(scoreList)
                if crossVal > bestScore:
                    bestModel, bestCV = model, crossVal
            except:
                continue
        return bestModel if bestModel else self.base_model(self.n_constant)
