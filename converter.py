import tensorflowjs as tfjs
from tensorflow.keras.models import model_from_json

TYPE_OF_DATA_AND_MODEL = 'vector'

#load json and create model
json_file = open(
    'models/model_{}_batchnorm_194.json'.format(TYPE_OF_DATA_AND_MODEL),
    'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(
    "models/model_{}_batchnorm_194.h5".format(TYPE_OF_DATA_AND_MODEL))
print("Loaded model from disk")

tfjs.converters.save_keras_model(model, 'webapp')

