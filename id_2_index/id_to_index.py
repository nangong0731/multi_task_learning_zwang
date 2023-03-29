# define a hash function to transfer an id to index, which will be used in embedding layer
import math
import numpy as np

def get_hash_number(id, seed1 = 2003, seed2 = 2011):
    '''
    :param id: the id need to be transfered, we hope these ids' type can be int, otherwise we use int() to change their type
    :param seed1: a prime number, which is supposed to bigger than the size of the ids' set
    :param seed2: another prime number bigger than seed1
    :return: index for each ID
    '''
    def get_hash_list(id, seed1 = 2003, seed2 = 2011):
        if type(id) != int:
            num = int(id)
        else:
            num = id

        res = []

        res.append(num % seed1)
        res.append(num % seed2)

        return res

    id_list = get_hash_list(id, seed1, seed2)

    hash_cycle = np.array(id_list) / math.sqrt(id_list[0] ^ 2 + id_list[1] ^ 2)
    hash_angle = math.atan(hash_cycle[1] / hash_cycle[0])

    return hash_angle