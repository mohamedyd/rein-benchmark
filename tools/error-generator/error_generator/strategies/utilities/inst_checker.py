import os
import zipfile
import requests
from gensim.models import KeyedVectors
import statistics
import numpy as np
from collections import Counter
import urllib
import urllib.request

class Factory(object):
    def __init__(self):
        self.called = False

    def get(self, input_class,dataset):
        similar_first=Similar_First()

        if not(self.called) and type(similar_first) is type(input_class):

            #this part occurs only once for all instances
            all_temp=similar_first.similar_first(dataset)
            self.called = True
            return all_temp

    def get_class(self,input_class,dataset):
        download = Download()
        if not (self.called) and type(download) is type(input_class):
           download.download("wiki-news-300d-1M.vec.zip","https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip",681808098)
           self.called = True


class Download(object):

    def download(self,filename, url, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""

        if not os.path.exists("../../datasets/wiki-news-300d-1M.vec.zip"):
            print("the required file doesn't exist in datasets folder so the system wants to download it for you")
            print("System try to download the wiki-news-300d-1M.vec.zip and put it in datasets folder")
            print("https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip")
            print("Downloading ...")
            filename, _ = urllib.request.urlretrieve(url , "../../datasets/"+filename)
            print("your download is finished")

        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('The required file found and verified', filename)

            if not os.path.exists("../../datasets/wiki-news-300d-1M.vec"):
                print("Unzipping ...")
                zip_ref = zipfile.ZipFile("../../datasets/wiki-news-300d-1M.vec.zip", 'r')
                zip_ref.extractall("../../datasets")
                zip_ref.close()
                print("Unziping is finished")
        else:
            print(statinfo.st_size)
            print("https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip")
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')

        return filename



class Load(object):
    def load(self):
        # Creating the model
        self.en_model = KeyedVectors.load_word2vec_format('../../datasets/wiki-news-300d-1M.vec')

        # Getting the tokens
        words = []
        for word in self.en_model.vocab:
            words.append(word)


class Similar_First(object):
    #this class used in similar_based_active_domain and implicit_missing_value

    def similar_first(self,dataset):
        temp = []
        all_temp=[]
        self.mod_value = []
        self.median_value = []

        for i in range(len(dataset[0])):
            for j in range(len(dataset)-1):
                temp.append(dataset[j][i])
            all_temp.append(temp)

            data = Counter(temp)
            uniqe = data.most_common()  # Returns all unique items and their counts
            mod = data.most_common(1)  # Returns the highest occurring item
            self.mod_value.append(mod[0][0])

            if (isinstance(dataset[1][i], str)):
                self.median_value.append(dataset[int(len(dataset) / 2)][i])
            else:
                self.median_value.append(statistics.median(temp))
            temp = []

        return all_temp

