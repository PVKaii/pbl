from keras import preprocessing
from Category import names
from Category import BASEDIR


trainGen=preprocessing.image.ImageDataGenerator(rescale=1./255)
valGen=preprocessing.image.ImageDataGenerator(rescale=1./255)
testGen=preprocessing.image.ImageDataGenerator(rescale=1./255)

trainBatches=trainGen.flow_from_directory(
    directory=BASEDIR+ 'train',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=50,
    shuffle=True,
    color_mode='rgb',
    classes=names
)

valBatches=valGen.flow_from_directory(
    directory=BASEDIR+'val',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=50,
    shuffle=False,
    color_mode='rgb',
    classes=names
)
testBatches=testGen.flow_from_directory(
    directory=BASEDIR+'test',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=4,
    shuffle=False,
    color_mode='rgb',
    classes=names
)
