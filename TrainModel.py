from Preprocessing import trainBatches
from Preprocessing import valBatches
from Preprocessing import testBatches
from keras import losses
from keras import models
import tensorflow as tf
from Category import BASEDIR

epochs=10
model=models.load_model(BASEDIR+'model/')
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=losses.categorical_crossentropy,
    metrics=['accuracy']
    )

history=model.fit(trainBatches,validation_data=valBatches,epochs=epochs)
model.evaluate(testBatches)
model.save(BASEDIR+'model/')