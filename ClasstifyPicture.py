from PIL import Image
from Category import names
from Category import categoryFolder
import shutil
import os
import tensorflow as tf

BASEDIR="E:/Python workspace/AI/AI/imageTest2/"
DATADIR="E:/Python workspace/AI/AI/image/"

tf.random.set_seed(1)

if not os.path.isdir(BASEDIR+'train/'):
    for name in names:
        os.makedirs(BASEDIR+"train/"+name)
        os.makedirs(BASEDIR+"val/"+name)
        os.makedirs(BASEDIR+"test/"+name)


# ----------------------------------------------------------------


for folderIdx,folder in enumerate(categoryFolder):
    files=os.listdir(DATADIR+folder)
    numberImages=len([name for name in files])
    nTrain=int((numberImages*0.6)+0.5)
    nValid=int((numberImages*0.25)+0.5)
    nTest=numberImages-nTrain-nValid
    print(numberImages,nTrain,nValid,nTest)
    for idx,file in enumerate(files):
        fileName=DATADIR+folder+file
        image=Image.open(fileName).resize((256,256))
        file=file.replace('jfif','jpg')
        if 'jpeg' in file:
            file=file.replace('jpeg','jpg')
            image= image.convert('RGB')
        if 'webp' in file:
            continue
        if idx<nTrain:
            # shutil.move(fileName,BASEDIR+'train/'+names[folderIdx])
            image.save(BASEDIR+'train/'+names[folderIdx]+'/'+file)
        elif idx < nTrain+nValid:
            # shutil.move(fileName,BASEDIR+'val/'+names[folderIdx])
            image.save(BASEDIR+'val/'+names[folderIdx]+'/'+file)
        else :
            # shutil.move(fileName,BASEDIR+'test/'+names[folderIdx])
            image.save(BASEDIR+'test/'+names[folderIdx]+'/'+file)