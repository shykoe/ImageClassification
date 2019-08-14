import os
import shutil
classdict={}
for root, dirs, files in os.walk('./101_ObjectCategories'):
        #print(root,dirs,files)
        if len(dirs) != 0:
                continue
        classname = root.split('/')[-1]
        classdict[classname] = []
        for i in files:
                assert os.path.exists(os.path.join(root,i))
                classdict[classname].append(os.path.join(root,i))
for key in classdict.keys():
        trainsplit = int(len(classdict[key])*0.8)
        #copy train
        path = './train/'+key
        os.mkdir(path)
        print('train')
        for i in classdict[key][:trainsplit]:
                shutil.copy(i,os.path.join(path,os.path.basename(i)))
        #copy val
        path = './val/'+key
        os.mkdir(path)

        for i in classdict[key][trainsplit:]:
                shutil.copy(i,os.path.join(path,os.path.basename(i)))
