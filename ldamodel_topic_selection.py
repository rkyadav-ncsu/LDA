"""
Modelling hollywood's actor's early life data for topic selection. The data is taken from early life section on wikipedia for each actor.
The theme of the topics are family, eduction, relationship.

all the files contains early childhood details of actors, except last one which contains details of humanitarian crisis in southern africa


"""
from spacy.lang.en import English
parser = English()

## imports for natural language toolkit

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('stopwords')  #filters out stop words for data cleaning
nltk.download('averaged_perceptron_tagger') ## to remove tags.
en_stop = set(nltk.corpus.stopwords.words('english'))

import gensim
from gensim.corpora import Dictionary

input_file_path= 'D:/ML/kaggle/'


# run assert test and print error message
def run_test(condition,message):
    assert condition, message

# open file, read all the lines and return them as text
def read_file(file_path):
    f = open(file_path,'rb')
    text = str(f.readlines())
    return text


##tokenize the string and return token array
def tokenize(text_to_tokenize):
    return_tokens = []
    #use parser to tokenize english language words.
    tokens = parser(text_to_tokenize)
    for token in tokens:
        if not token.orth_.isspace():
            return_tokens.append(token.lower_)
    return return_tokens


def morph_words(word_to_morph):
    morphed_word = wn.morphy(word_to_morph)
    if morphed_word is None:
        return word_to_morph
    else:
        return morphed_word

def get_token_for_lda(text):
    tokens = tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [morph_words(token) for token in tokens]
    ##assign pos tags to remove NNP and NNPS
    tagged_tokens = nltk.tag.pos_tag(tokens)
    edited_tokens = [word for word, tag in tagged_tokens if tag != 'NNP' and tag != 'NNPS']


    return edited_tokens



######### LDA #########

def lda_model():

    token_arr = []
    for i in range(1,11):
        file_name = input_file_path+'data_'+str(i)+'.txt'
        data =  read_file(file_name)
        run_test(data !='',file_name+' file is empty')

        token_arr.append(get_token_for_lda(data))
    ## create corpora dictionary
    dictionary = Dictionary(token_arr)
    ## convert document to bag of words.
    corpus = [dictionary.doc2bow(text) for text in token_arr]


    number_of_topics = 6
    passes = 1000
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = number_of_topics
                                               , id2word=dictionary, passes=passes)

    ## select top three words and print.
    topics = ldamodel.print_topics(num_words=3)
    topics_dict =dict()
    for (index,topic) in topics:
        print(str(index)+' : '+topic)
        topics_dict[index] = topic
    return ldamodel,topics_dict


def lda_test(ldamodel,topics_dict,file_index):
    ### testing on document 11,12,13,14
    test_token_arr = []
    file_name = input_file_path + 'data_' + str(file_index) + '.txt'
    test_data = read_file(input_file_path+'data_'+str(file_index)+'.txt')
    run_test(test_data  != '', file_name + ' file is empty')
    test_token_arr.append(get_token_for_lda(test_data))


    dictionary= Dictionary(test_token_arr)
    new_doc_bow = dictionary.doc2bow(test_token_arr[0])
    print()
    print('---------'+file_name+'--------')
    result = ldamodel.get_document_topics(new_doc_bow)
    # sorted(result)
    for (index,value) in result:
        print('prob:'+str(value) +' -> '+topics_dict[index])



if __name__=='__main__':
    ##train the model
    ldamodel,topics_dict=lda_model()
    ##use the model to test on file 11,12,13.
    lda_test(ldamodel, topics_dict,11)
    lda_test(ldamodel, topics_dict,12)
    lda_test(ldamodel, topics_dict,13)
    lda_test(ldamodel, topics_dict,14)