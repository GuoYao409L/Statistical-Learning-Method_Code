import os
import random


def load(path):
    data = []
    fr = open(path, 'r', encoding='utf-8')
    for line in fr.readlines():
        data.append(line.strip())
    return data

def save(data, path):
    fw = open(path, 'w', encoding='utf-8')
    for line in data:
        fw.write(line + '\n')
    fw.close()

if __name__ == '__main__':
    type = 'test'
    data = load('0.datasets/Mnist/mnist_{}.txt'.format(type))
    random.shuffle(data)

    data = data[:100]
    save(data, '0.datasets/Mnist/mnist_{}_new.txt'.format(type))