from keras import models
from PIL import Image
import numpy as np
from Category import BASEDIR
from Category import names
src=BASEDIR+'rauma.jpg'

model=models.load_model(BASEDIR+'/model')

image=Image.open(src)
image=image.resize((256,256))
image=np.array(image)/255.0
predict=model.predict(image.reshape(1,256,256,3))
print(names[predict.argmax()])
