from .utilities import *
import random
from random import randint
from random import shuffle
import nltk
from nltk.corpus import stopwords
import yaml
from yaml import Loader

### download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
###########################


class TextProcess:
    """
    Object for converting text (name set) to sparse matrix    
    """
    
    def __init__(self, language_letters: str, gender_dict: dict, 
                 params: dict, loggingpath = '/', 
                 size_patch = 60000, array_size = 54):
        
        self.letters = language_letters
        self.size_letters = len(language_letters)
        self.genderdict = gender_dict
        self.logging_path = loggingpath
        self.params = params
        self.size_max = array_size
        self.location = 'text_processing'
        self.size_patch = size_patch
        self._get_init_logging()
    
    
    def _get_init_logging(self):    
        """
        logging
        """
        
        self.logger = CustomLOG(
            format_str = FORMAT,
            log_path = os.path.join(self.logging_path, f'{self.location}.log'),
            resource = self.location,
        )
        
    
    def sparse_array(self, pos, size_array):
        """
        making array with 0.0 and label 1.0 at the position pos
        """
    
        out_array = [0.0] * size_array
        out_array[int(pos)] = 1.0
        
        return out_array
    
    
    def index_of_letter(self, l):
        """
        definition the letter's position in ABCDEFGHIJKLMNOPQRSTUVWXYZ 
        """
    
        l = l.upper()
        if (l not in self.letters):
            return 0
        else:
            return self.letters.index(l)
    
    
    def word_to_array(self, word):
        """
        making sparse matrix with size count of letters X dictsize from the word
        """
    
        word = word.upper()
        word = re.sub('[^A-ZА-Я ]+', '', word)
        
        word_array = list(map(lambda x: self.sparse_array(self.index_of_letter(x), self.size_letters), word.upper().replace(" ", "")))
        word_array += [self.sparse_array(self.size_letters - 1, self.size_letters)] * (self.size_max - len(word_array))
    
        return list(word_array)
    
    
    def list_of_words_to_array(self, wordlist):
        """
        making sparse matrix with size count of letters X dictsize from the list of words
        """
        
        output = []
        for word in wordlist:
            if word == '': continue
            
            word = word.upper()
            word = re.sub('[^A-ZА-Я ]+', '', word)
            word = word[: self.size_max]
            word_array = list(map(lambda x: self.sparse_array(self.index_of_letter(x), self.size_letters), word.upper().replace(" ", "")))
            word_array += [self.sparse_array(self.size_letters - 1, self.size_letters)] * (self.size_max - len(word_array))
            output.append(list(word_array))
            
        return output
    
    
    def make_batch(self, data, language_stopwords = 'russian'):
        """
        processing array of words to sparse array of indexes after shuffle
        """
        
        self.logger.info(f"Preparing batch with size: {self.size_patch} and constraint of word-array: {self.size_max} ...", kind = 'preparing data')
    
        max_input, counter, gender_out, word_input, check_count = 0, 0, [], [], [0, 0]
        
        stop_words = stopwords.words(language_stopwords)
        
        ### Shuffle
        shuffle(data)
        
        while counter <= self.size_patch:
            if counter >= len(data) - 1:
                self.logger.warning(f"Counter - {counter} out of data length - {len(data)}. Break!", kind = 'preparing data')
                break
            name = data[counter]
            ## name[0] = first name
            ## name[1] = last_name
            ## name[2] = middle_name
            ## name[3] = gender (M, F)
            if not all(name):
                counter += 1
                continue
            elif (name[3].strip() not in self.genderdict):
                counter += 1
                continue
            
            max_input = max(max_input, len(name[0]))
            random_mix_kind_1 = randint(0, 10)     ## mixing of correct and uncorrect namesets
            random_mix_kind_2 = randint(0, 1)      ## mixing part of names inside namesets
            
            if random_mix_kind_1 < 7:
                check_count[0] += 1
                
                if random_mix_kind_1 == 0:
                    name_set = name[1].strip()
                elif random_mix_kind_1 < 4:
                    if random_mix_kind_2 == 0:
                        name_set = f"{name[0]} {name[1]} {name[2]}".strip()
                    else:
                        name_set = f"{name[1]} {name[0]} {name[2]}".strip()
                elif random_mix_kind_1 < 6:
                    if random_mix_kind_2 == 0:
                        name_set = f"{name[0]} {name[1]}".strip()
                    else:
                        name_set = f"{name[1]} {name[0]}".strip()
                else:
                    if random_mix_kind_2 == 0:
                        name_set = f"{name[0]} {name[2]}".strip()
                    else:
                        name_set = f"{name[2]} {name[0]}".strip()
                if len(name_set) < self.size_max:
                    word_input.append(self.word_to_array(name_set))
                    gender_out.append(self.sparse_array(self.genderdict[name[3].strip()], len(self.genderdict)))
                counter += 1
            else:
                check_count[1] += 1
                if random_mix_kind_1 == 7:
                    name_set = random.choice(stop_words)
                elif random_mix_kind_1 == 8:
                    name_set = f"{random.choice(stop_words)} {random.choice(stop_words)}"
                elif random_mix_kind_1 == 9: 
                    # Random string
                    name_set = ''.join(random.choices(self.letters, k = randint(3, 18)))
                else:
                    name_set = f"{random.choice(stop_words)} {random.choice(stop_words)} {random.choice(stop_words)}"
                
                name_set = name_set.strip().upper()
                if len(name_set) < self.size_max:
                    word_input.append(self.word_to_array(name_set))
                    gender_out.append(self.sparse_array(self.genderdict["unknown"], len(self.genderdict)))
        
        
        self.logger.info(f"Processed {check_count[0]} input data and {check_count[1]} synthetic (random) data!", kind = 'preparing data')    
        x = np.array(word_input).astype('float32')
        proporcions = [int(0.75 * len(x)), len(x)]
        x_train, x_test, _ = np.split(x, proporcions)
        action_train, action_test, _ = np.split(gender_out, proporcions)
        
        self.logger.info(f"Made {len(x_train)} training dataset and {len(x_test)} testing dataset.", kind = 'preparing data')

        return x_train, x_test, action_train, action_test
    
    
    def make_predict(self, model = None, names_list = []):
        """
        the input is model and names_list 
        the output is dictionary with result and their profitability
        """
        
        result_dictionary = {}
        
        if names_list == []: return result_dictionary
    
        sparse_names = self.list_of_words_to_array(names_list)
        
        try:
            predict_list = model.predict(np.array(sparse_names))
        except Exception as err:
            self.logger.error(f"Problem with prediction process : Lenght of array - {len(names_list)}, Error - {err}(", kind = 'predict data')
            return result_dictionary
        
        genderdict_reverse = {v:k for k,v in self.genderdict.items()}
        
        for i, name in enumerate(names_list):
            
            result_dictionary[i] = {}
            index_max = np.argmax(predict_list[i])
            result_dictionary[i]['name'] = name
            result_dictionary[i]['gender'] = genderdict_reverse[index_max]
            
            for k,v in self.genderdict.items():
                result_dictionary[i][f'probability_{k}'] = np.round(predict_list[i][v], 4)
        
        return result_dictionary
    

#################################################################################################################################################