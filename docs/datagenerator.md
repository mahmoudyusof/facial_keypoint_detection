# How to use data generators in tensorflow

## Why ?

Believe it or not, but loading the entire dataset in memory is **NOT** the best idea.  
If you're dealing with a small dataset, that might work, but that is just a waste of resources, and worse if you're working on a huge dataset like the imageNet dataset, this won't work at all.

## HOW ?

Python generators are lazy which means they are iterables that give you the data upon request, unlike regular lists that just store the data in memory all the time.

tensorflow keras has a `Sequence` class that can be used for this purpose.  
let's jump into it

## Scenrio

working with images is a good example for this, so let's say that you have pictures of objects that you need to localize, so your features are images and labels are (x, y, h, w) for coordinate and dimensions of the containing box with the labels and image names are stored in a csv file.

## Code

let's define an initializer

```python
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
  """
    Args:
      csv_file: file in which image names and numeric labels are stored
      base_dir: the directory in which all images are stored
      output_size: image output size after preprocessing
  """
  def __init__(self, csv_file, base_dir, output_size, shuffle=False, batch_size=10):
    self.df = pd.read_csv(csv_file)
    self.base_dir = base_dir
    self.output_size = output_size
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.on_epoch_end()
```

Now let's define some special methods starting with the one called in the initializer `on_epoch_end()` that is called after each epoch as the name may suggest, duh!

```python
def on_epoch_end(self):
  # just create an array of indecies
  # for all of the entries in the dataset
  self.indecies = np.arange(len(self.df))
  if self.shuffle:
    np.random.shuffle(self.indecies)
```

Now we need to define the length of the data, which is not the number of entries as you might think, it's actually the number of batches, this needs to be accessible with the `len` function in python so we need to define the `__len__` method.

```python
def __len__(self):
  return int(len(self.df) / self.batch_size)
```

Now let's get serious, the fun part is in the next method which is `__getitem__`.  
This function gets called on indexing or slicing like:

```python
generator[0]
generator[3:5]
```

and in this function we shall load and preprocess the images.  
You might think splitting this into multiple functions would be a good idea ... and you'd totally be right.  
This fucntion should return a preprocessed batch of data

```python
def __getitem__(self):
  ## Initializing Batch
  #  that one in the shape is just for a one channel images
  # if you want to use colored images you might want to set that to 3
  X = np.empty((self.batch_size, *self.output_size, 1))
  # (x, y, h, w)
  y = np.empty((self.batch_size, 4, 1))

  # get the indecies of the requested batch
  indecies = self.indecies[idx*self.batch_size:(idx+1)*self.batch_size]

  for index in range(len(indecies)):
    img_path = os.path.join(self.base_dir,
                self.df.iloc[indecies[index], 0])

    img = mpimg.imread()

    ## this is where you preprocess the image
    ## make sure to resize it to be self.output_size

    label = self.df.iloc[indecies[index], 1:].to_numpy()
    ## if you have any preprocessing for
    ## the labels too do it here

    X[index] = img
    y[index] = label

  return X, y
```

Now you are ready to train the model with this generator

```python
from tensorflow.keras.models import Sequential

model = Sequential([
  ## define the model's architecture
])

train_gen = DataGenerator("data.csv",
                          "data",
                          (244, 244),
                          batch_size=20,
                          shuffle=True)

## compile the model first of course

# now let's train the model
model.fit(train_gen, epochs=5, ...)
#  note you could also make a valdation generator and pass it here like normal datasets

# back in the days you had to do this
# model.fit_generator(train_gen, ...)
```

## The End

And that's it.. you've just created your dataset generator that loads the data into memory batch by batch instead of the whole thing at once.  
I hope this was useful.

[Reference Article](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

[My Notebooks](https://github.com/mahmoudyusof/facial_keypoint_detection)
