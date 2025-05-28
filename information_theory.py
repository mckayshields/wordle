"""
Information Theory Lab

Name McKay Shields
Section Vol 3
Date 9/26/23
"""


import numpy as np
import wordle
import random

# Problem 1
def get_guess_result(guess, true_word):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the secret word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        guess (string) - the guess being made
        true_word (string) - the secret word
    Returns:
        result (list of integers) - the result of the guess, as described above
    """
    guess=[*guess] #turn strings into lists
    true_word=[*true_word]
    result=np.zeros(len(guess))
    for i in range(len(guess)):
        if guess[i]==true_word[i]: #mark green
            result[i]=2
            true_word[i]=0
    for i in range(len(guess)):
        if guess[i]==2: #move on if already green
            pass
        if guess[i] in true_word: #mark yellow
            result[i]=1
            true_word[true_word.index(guess[i])]=0 #remove that letter from true word to avoid double counting
    return list(result) #return results


# Helper function
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words
    
# Problem 2
def compute_highest_entropy(all_guess_results, allowed_guesses):
    """
    Compute the entropy of each allowed guess.
    
    Arguments:
        all_guess_results ((n,m) ndarray) - the array found in
            all_guess_results.npy, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_guesses (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
    """
    n,m=np.shape(all_guess_results)
    entropies=np.zeros(n) #initialize vector of all entropies
    for i in range(n):
        guess=all_guess_results[i] #iterate through each column
        new_entropy=0
        values, counts=np.unique(guess,return_counts=True) #count each value in the column
        total=sum(counts) #sum for entropy calculation
        for j in range(len(values)):
            new_entropy-=counts[j]/total*np.log2(counts[j]/total) #calculate entropy
        entropies[i]=new_entropy
    return allowed_guesses[np.argmax(entropies)] #get best guess

# Problem 3
def filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already have an array of the result of all guesses for all possible words, 
    we will use this array instead of recomputing the results.
    
	Return a filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (2-D ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        allowed_guesses (list of str)
            The list of words we are allowed to guess
        possible_secret_words (list of str)
            The list of possible secret words
        guess (str)
            The guess we made
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (2-D ndarray) The filtered array of guess results
    """
    decimalresult=0
    guessindex=allowed_guesses.index(guess) #get index of guess
    for i in range(5):
        decimalresult+=result[i]*3**i #convert ternary list to decimal
    mask= all_guess_results[guessindex,:] == decimalresult #get all columns with decimalresult in column
    return [possible_secret_words[k] for k in range(len(mask)) if mask[k]==True], all_guess_results[:,mask] #use mask to filter possible words and guess result matrix

# Problem 4
def play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    while game.is_finished()==False:
        if len(possible_secret_words)==1: #if theres only one possible word, choose that
            guess=possible_secret_words[0]
        else:
            guess=random.choice(allowed_guesses) #else just make a random guess    
        result, guesscount=game.make_guess(guess)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result) #filter based on new result
    return guess, guesscount

# Problem 5
def play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    while game.is_finished()==False:
        if len(possible_secret_words)==1:
            guess=possible_secret_words[0] #if theres only one possible word, choose it
        else:
            guess=compute_highest_entropy(all_guess_results, allowed_guesses) #or guess word with but entropy
        result, guesscount = game.make_guess(guess)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result) #filter words that only are possible
    return guess, guesscount

# Problem 6
def compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    naivesum,entropysum=0,0 #initialize sums
    for i in range(n):
        game=wordle.WordleGame() #start game
        word,count=play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses) #get count for naive method
        naivesum+=count #sum over each iteration
        game=wordle.WordleGame() #start game
        word,count=play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses) #get guess count for entropy method
        entropysum+=count #sum over iterations
    return naivesum/n, entropysum/n #return averages

def testfunc():
    '''vibes=np.load('all_guess_results.npy')
    b=load_words('allowed_guesses.txt')
    return compute_highest_entropy(vibes,b)
    result=[0,0,0,2,1]
    m=np.array([[1,4,7],[6,6,7],[7,0,7]])
    return filter_words(m,['a','b','c'],['x','y','z'],'c',result)
    array=np.load('all_guess_results.npy')
    guesses=load_words('allowed_guesses.txt')
    secrets=load_words('possible_secret_words.txt')
    return filter_words(array, guesses, secrets, 'boxes', [0,0,0,2,1])[1].shape'''
    game=wordle.WordleGame()
    array=np.load('all_guess_results.npy')
    guesses=load_words('allowed_guesses.txt')
    secrets=load_words('possible_secret_words.txt')
    #return play_game_naive(game,array,secrets,guesses) #prob4
    #return play_game_entropy(game,array,secrets,guesses) #prob5
    return compare_algorithms(array, secrets, guesses, n=100) #prob6

def playwordle():
    all_guess_results=np.load('all_guess_results.npy')
    allowed_guesses=load_words('allowed_guesses.txt')
    possible_secret_words=load_words('possible_secret_words.txt')
    result=None
    n=0
    while result!=[2,2,2,2,2]:
        if len(possible_secret_words)==1:
            guess=possible_secret_words[0] #if theres only one possible word, choose it
        else:
            guess=compute_highest_entropy(all_guess_results, allowed_guesses) #or guess word with but entropy
        print('Reccomended guess:'+str(guess))
        guess=input('Enter your next guess: ')
        n+=1
        resultstring=input('Enter results of guess:'+'\n'+'Gray=0; Yellow=1; Green=2; \t')
        result=[int(i) for i in resultstring]
        print(result)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result) #filter words that only are possible
    print(f'Congratulations! You got Wordle in {n} guesses.')