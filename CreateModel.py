from Category import names
from keras import layers
from keras import models
from Category import BASEDIR


numberOutput=len(names)
numberConvLater=2
numberHidenLayer=2

# model=models.Sequential([    
#     layers.Conv2D(32,(3,3),activation='relu',input_shape=(256, 256, 3),strides=(1,1),padding='valid'),
#     layers.MaxPool2D(pool_size=(2,2)),
#     layers.Conv2D(64,3,activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64,activation='relu'),
#     layers.Dense(3,activation='softmax')    
# ])
# print(model.summary())
# model.save(BASEDIR)


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(256, 256, 3),strides=(1,1),padding='valid'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
for i in range(numberConvLater):
    model.add(layers.Conv2D(64,3,activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
for i in range(numberHidenLayer):
    model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(numberOutput,activation='softmax'))
print(model.summary())
model.save(BASEDIR+'model/')