#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/anshul/MAchine_Learning_code/Project/Input_img/"
path_save = "/home/anshul/MAchine_Learning_code/Project/Converted/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            g, h = os.path.splitext(path_save+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(g + '.png', 'PNG', quality=64)

resize()