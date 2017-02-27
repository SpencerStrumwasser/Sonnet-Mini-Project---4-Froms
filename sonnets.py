from HMM import unsupervised_HMM
from Utility import Utility
from nltk.corpus import cmudict
from numpy import random

# sonnet is list of sonnets, rhyme scheme contains corresponding sonnet line
# that current index rhymes with
def rhyme_time(sonnets, rhyme_scheme):
    rhyme_map = {}
    for k in range(int(len(sonnets) / 14)):
        sonnet = sonnets[14*k : 14*(k+1)]
        for i in range(len(rhyme_scheme)):
            word = sonnet[i][-1]
            rhyme = sonnet[rhyme_scheme[i]][-1]
            if word == rhyme:
                continue
            if word not in rhyme_map:
                rhyme_map[word] = [rhyme]
            else:
                if rhyme not in rhyme_map[word]:
                    rhyme_map[word].append(rhyme)
    return rhyme_map

# produces a dictionary containing counts for how often shakespeare uses words to end
# his sentences
def ending_words(sonnets):
    weights = {}
    for k in range(int(len(sonnets) / 14)):
        sonnet = sonnets[14*k : 14*(k+1)]
        for i in range(14):
            last_word = sonnet[i][-1]
            if last_word in weights:
                weights[last_word] += 1
            else:
                weights[last_word] = 1
    return weights

# given a list of ending words and ending probabilities, produces
# a pair of rhyming words to end two sentences
def generate_ending_rhymes(end_words, end_probs, rhyme_map):

    final_end_words = [0 for i in range(14)]
    for i in range(4):
        if i != 3:
            word1 = random.choice(end_words, p=end_probs)
            word2 = random.choice(rhyme_map[word1])
            word3 = random.choice(end_words, p=end_probs)
            word4 = random.choice(rhyme_map[word3])
            final_end_words[4 * i] = word1
            final_end_words[4 * i + 1] = word3
            final_end_words[4 * i + 2] = word2
            final_end_words[4 * i + 3] = word4
        elif i == 3:
            word1 = random.choice(end_words, p=end_probs)
            word2 = random.choice(rhyme_map[word1])
            final_end_words[4 * i] = word1
            final_end_words[4 * i + 1] = word2
    return final_end_words


def unsupervised_learning(n_states, n_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    lines, word_map = Utility.load_shakespeare_hidden()
    number_map = {word:num for num,word in word_map.items()}

    rhyme_scheme = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 13, 12]
    rhyme_map = rhyme_time(lines, rhyme_scheme)
    
    end_word_counts = ending_words(lines)
    end_words = []
    end_counts = []
    # split our counts into two separate lists
    for word, count in end_word_counts.items():
        end_words.append(word)
        end_counts.append(count)
    end_probs = [end_counts[i] / len(lines) for i in range(len(end_counts))]

    final_end_words = generate_ending_rhymes(end_words, end_probs, rhyme_map)

    # Train the HMM.
    HMM = unsupervised_HMM(lines, n_states, n_iters)
    # print out most frequent words for each state
    # HMM.print_frequent_words(number_map)
    sonnet = []
    for i in range(14): 
        sonnet.append(HMM.generate_emission(10, final_end_words[i], number_map))

    for line in sonnet:
        trans_line = " ".join(number_map[int(elem)] for elem in line)
        print(trans_line)

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Training on sonnets"))
    print('#' * 70)
    print('')
    print('')

    unsupervised_learning(8, 30)