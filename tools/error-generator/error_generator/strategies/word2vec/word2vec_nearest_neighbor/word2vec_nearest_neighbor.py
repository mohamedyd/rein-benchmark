from operator import itemgetter
from error_generator.strategies.utilities.inst_checker import Factory,Download,Load



class Word2vec_Nearest_Neighbor():
    def __init__(self,name="Word2vec_Nearest_Neighbor"):
        self.name=name
        self.called = False

    def run(self,row,col,selected_value,dataset):
        similar=[]

        if self.called == False:
            ins_download = Download()
            f = Factory()
            f.get_class(ins_download, dataset)
            l=Load()
            l.load()
            self.called = True

        l = Load()
        l.load()
        # Pick a word
        find_similar_to = selected_value

        # Finding out similar words [default= top 10]
        if find_similar_to in l.en_model.vocab:

            for similar_word in l.en_model.similar_by_word(find_similar_to):
                # print("Word: {0}, Similarity: {1:.2f}".format(similar_word[0], similar_word[1]))
                similar.append([similar_word[0], similar_word[1]])
            list_similar = sorted(similar, key=itemgetter(1), reverse=True)
            print(similar)
        else:
            print("the selscted value doesn't have any similar in our wiki-news-300d-1M.vec")
            list_similar=[[0]]
            list_similar[0][0]=""

        return list_similar[0][0]