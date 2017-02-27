# Loads in the appropriate datasets
import string

class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__():
        pass

    @staticmethod
    def load_shakespeare():
        '''
        Loads the file 'shakespeare.txt'.

        Returns:
            moods:      Sequnces of states, i.e. a list of lists.
                        Each sequence represents half a year of data.
            mood_map:   A hash map that maps each state to an integer.
            genres:     Sequences of observations, i.e. a list of lists.
                        Each sequence represents half a year of data.
            genre_map:  A hash map that maps each observation to an integer.
        '''
        lines = []
        word_map = {}
        word_counter = 0
        line_counter = 0
        # we want to skip these sonnets since they do not follow the usual sonnet outline
        sonnet_99 = list(range(1667, 1683, 1))
        sonnet_126 = list(range(2127, 2140, 1))
        # load in dictionary of python punctuation characters to remove
        remove_punctuation = dict.fromkeys(map(ord, ',:;.?!()'))
        # splitting data by line
        with open("shakespeare.txt") as f:
            for line in f:
                line_counter += 1
                # skipping these two sonnets
                if line_counter in sonnet_99 or line_counter in sonnet_126:
                    continue
                # indicates number labelling sonnet
                if line.strip().isdigit():
                    continue
                elif line == '\n':
                    continue

                # get rid of punctuation/spaces and split into words
                temp_line = line.translate(remove_punctuation).split()
                # builds our mapped words
                new_line = []
                for word in temp_line:
                    if word.lower() not in word_map:
                        word_map[word.lower()] = word_counter
                        word_counter += 1

                    new_line.append(word_map[word.lower()])

                lines.append(new_line)

        return lines, word_map

    @staticmethod
    def load_shakespeare_hidden():
        '''
        Loads the file 'shakespeare.txt' and hides the states.

        Returns:
            genres:     The observations.
            genre_map:  A hash map that maps each observation to an integer.
        '''
        lines, word_map = Utility.load_shakespeare()
        return lines, word_map
