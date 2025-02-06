# this code is written by Tonmoy Ghosh (tghosh@crimson.ua.edu)
# this code will copy the image files and create the dataset for NoisyViT_food

import os.path
import shutil


#for food2k, download the food2k dataset and run this code
# change the srcprefix and dstprefix accordingly
srcprefix = r'C:\food database\food2k\images/'  # source prefix
dstprefix = r'C:\NoisyViT_food\food2k\train/'  # destination prefix
f = open(r"C:\food database\food2k\meta_data\train_finetune.txt", "r")
print(f.readline())
#linetext = f.readline()
for linetext in f:
    temp = linetext.split()
    #print(temp)
    if os.path.exists(dstprefix+temp[1]):
        shutil.copyfile(srcprefix + temp[0],dstprefix + temp[0])
    else:
        os.mkdir(dstprefix + temp[1])
        shutil.copyfile(srcprefix + temp[0],dstprefix + temp[0])
f.close()

srcprefix = r'C:\food database\food2k\images/'  # source prefix
dstprefix = r'C:\NoisyViT_food\food2k\val/'  # destination prefix
f = open(r"C:\food database\food2k\meta_data\val_finetune.txt", "r")
print(f.readline())
#linetext = f.readline()
for linetext in f:
    temp = linetext.split()
    #print(temp)
    if os.path.exists(dstprefix+temp[1]):
        shutil.copyfile(srcprefix + temp[0],dstprefix + temp[0])
    else:
        os.mkdir(dstprefix + temp[1])
        shutil.copyfile(srcprefix + temp[0],dstprefix + temp[0])
f.close()

srcprefix = r'C:\food database\food2k\images/'  # source prefix
dstprefix = r'C:\NoisyViT_food\test/'  # destination prefix
f = open(r"C:\food database\food2k\meta_data\test_finetune.txt", "r")
print(f.readline())
#linetext = f.readline()
for linetext in f:
    temp = linetext.split()
    #print(temp)
    if os.path.exists(dstprefix+temp[1]):
        shutil.copyfile(srcprefix + temp[0],dstprefix + temp[0])
    else:
        os.mkdir(dstprefix + temp[1])
        shutil.copyfile(srcprefix + temp[0],dstprefix + temp[0])
f.close()

#for food101, download the food101 dataset and run this code
# change the srcprefix and dstprefix accordingly
'''
srcprefix = r'C:\food database\food-101\images/' # source prefix
dstprefix = r'C:\NoisyViT_food\food101\train/'  # destination prefix
f = open(r"C:\food database\food-101\meta\train.txt", "r")
print(f.readline())
#linetext = f.readline()
for linetext in f:
    temp = linetext.split('/')
    #print(temp)
    fname = linetext[:-1]+'.jpg'    #removing newline
    if os.path.exists(dstprefix+temp[0]):
        shutil.copyfile(srcprefix + fname,dstprefix + fname)
    else:
        os.mkdir(dstprefix + temp[0])
        shutil.copyfile(srcprefix + fname, dstprefix + fname)
f.close()

srcprefix = r'C:\food database\food-101\images/'
dstprefix = r'C:\NoisyViT_food\food101\val/'
f = open(r"C:\food database\food-101\meta\test.txt", "r")
print(f.readline())
#linetext = f.readline()
for linetext in f:
    temp = linetext.split('/')
    #print(temp)
    fname = linetext[:-1]+'.jpg'    #removing newline
    if os.path.exists(dstprefix+temp[0]):
        shutil.copyfile(srcprefix + fname,dstprefix + fname)
    else:
        os.mkdir(dstprefix + temp[0])
        shutil.copyfile(srcprefix + fname, dstprefix + fname)
f.close()
'''