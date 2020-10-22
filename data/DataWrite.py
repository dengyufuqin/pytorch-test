import os
import copy

def write_txt(path, txt_path):
    num = len(os.listdir(path))
    file_path = txt_path

    file = open(file_path, 'w')
    Y = 0
    c = os.listdir(path)

    for categray in c:
        C = c.index(categray)
        for imgs in os.listdir(os.path.join(path, categray)):
            file.write(os.getcwd()+'/'+path+categray+'/'+imgs+'|'+str(Y)+'\n')
        Y = Y + 1

if __name__ == '__main__':
    write_txt('train/',
              'train.txt')

    write_txt('test/',
              'test.txt')

