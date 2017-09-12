# begin
import numpy as np
import time

ratings = []
movies = []
users = []

# TODO: Implement two methods, one filling the matrix on-the-fly, the other first loading everything into memory

def read_data_set_into_memory():


# loads the ml-1m dataset and returns a triple (ratings, movies, users)
def readDataset():
    start_time = time.time()

    # load the ratings dataset
    ratingsFile = open('ml-1m/ratings.dat', 'r', encoding='latin-1')
    for line in ratingsFile:
        ratings.append(convertFileEntryToTuple(line))
    ratingsFile.close()

    stop_time1 = time.time()
    print('Loading the ratings dataset too: ' + str(stop_time1 - start_time) + ' seconds')

    # load the movies dataset
    moviesFile = open('ml-1m/movies.dat', 'r', encoding='latin-1')
    for line in moviesFile:
        movies.append(convertFileEntryToTuple(line))
    moviesFile.close()

    stop_time2 = time.time()
    print('Loading the movies dataset too: ' + str(stop_time2 - stop_time1) + ' seconds')

    # load the users dataset
    usersFile = open('ml-1m/users.dat', 'r', encoding='latin-1')
    for line in usersFile:
        users.append(convertFileEntryToTuple(line))
    usersFile.close()

    stop_time3 = time.time()
    print('Loading the users dataset too: ' + str(stop_time3 - stop_time2) + ' seconds')
    print('Loading data in total took ' + str(stop_time3 - start_time) + ' seconds')


    # all data sets loaded into normal python lists (maybe this will be very slow)
    # return ratings, movies, users


# takes a FileEntry as a single string as input and transforms it into a tuple of ints
def convertFileEntryToTuple(entry):
    stringlist = entry.split("::")
    return tuple(map(parseStringToInt, stringlist))


def parseStringToInt(string):
    returnValue = 0
    try:
        returnValue = int(string)
    except (ValueError, UnicodeDecodeError):
        returnValue = string
    finally:
        return returnValue


'''
### first recommender system steps ###

Todo:
    1. get average rating of other users for an item
'''


def getUserRatings(userId):
    result_list = []
    for rating in ratings:
        if (rating[0] == userId):
            result_list.append(rating)
    return result_list


def getMovieRatings(movieId):
    result_list = []
    for rating in ratings:
        if (rating[1] == movieId):
            result_list.append(rating)
    return result_list


def getMeanMovieRating(movieId):
    ratings = getMovieRatings(movieId)
    # [(1,3,5,19173459),etc] (UserID,MovieID,Rating,Timestamp)
    mean_sum = 0
    for tpl in ratings:
        mean_sum += tpl[2]


readDataset()
# end