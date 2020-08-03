# How to use data generators in tensorflow

---

## Why ?

Believe it or not, but loading the entire dataset in memory is **NOT** the best idea.  
If you're dealing with a small dataset, that might work, but that is just a waste of resources, and worse if you're working on a huge dataset like the imageNet dataset, this won't work at all.

## HOW ?

Python generators are lazy which means they are iterables that give you the data upon request, unlike regular lists that just store the data in memory all the time.

tensorflow keras has a `Sequence` class that can be used for this purpose. <a class="mdlink" href="https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence">Sequence Class API Reference</a>  
let's jump into it

## Scenario

Working with images is a good example for this, so let's say that you have pictures of objects that you need to localize,  
So your features are images and labels are (x, y, h, w) for coordinate and dimensions of the containing box, and the labels and image names are stored in a csv file.

## Data

| image_file &nbsp; &nbsp; | x &nbsp; &nbsp; | y &nbsp; &nbsp; | w &nbsp; &nbsp; | h &nbsp; &nbsp; |
| ------------------------ | --------------- | --------------- | --------------- | --------------- |
| file1.png                | 10              | 20              | 50              | 50              |
| ...                      | ...             | ...             | ...             | ...             |
| ...                      | ...             | ...             | ...             | ...             |

---

## Code

Let's define an initializer, the initializer is going to take the information needed to get the data such as:

- The csv file
- The directory containing all of the images

It will also take the output shape of the batch

- The output size of each image
- The batch size

```python

import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):

  def __init__(self, csv_file, base_dir, output_size, shuffle=False, batch_size=10):
  """
  Initializes a data generator object
    :param csv_file: file in which image names and numeric labels are stored
    :param base_dir: the directory in which all images are stored
    :param output_size: image output size after preprocessing
    :param shuffle: shuffle the data after each epoch
    :param batch_size: The size of each batch returned by __getitem__
  """
    self.df = pd.read_csv(csv_file)
    self.base_dir = base_dir
    self.output_size = output_size
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.on_epoch_end()


```

---

Now let's define some special methods starting with the one called in the initializer `on_epoch_end()` that is called after each epoch as the name may suggest, duh!

We call this method in the initializer because we need the indeces attribute to be set at the begining of the first epoch, otherwise we will get an error telling us that the class has no attribute "indecies"

```python

def on_epoch_end(self):
  self.indices = np.arange(len(self.df))
  if self.shuffle:
    np.random.shuffle(self.indices)


```

---

Now we need to define the length of the data, which is not the number of entries as you might think, it's actually the number of batches, this needs to be accessible by the `len` function in python so we need to define the `__len__` method.

```python

def __len__(self):
  return int(len(self.df) / self.batch_size)


```

---

Now let's get serious, the fun part is in the next method which is `__getitem__`.  
This function gets called on indexing or slicing like `data_generator[0]` or `data_generator[1:3]` and the index is passed as a parameter to it. Here we call it `idx`

In this function we shall load and preprocess the images.  
This will only be fired when keras trys to load a batch, which will save our memory.

You might think splitting this into multiple functions would be a good idea ... and you'd be totally right.  
This function should return a preprocessed batch of data

```python

def __getitem__(self, idx):
  ## Initializing Batch
  #  that one in the shape is just for a one channel images
  # if you want to use colored images you might want to set that to 3
  X = np.empty((self.batch_size, *self.output_size, 1))
  # (x, y, h, w)
  y = np.empty((self.batch_size, 4, 1))

  # get the indices of the requested batch
  indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

  for i, data_index in enumerate(indices):
    img_path = os.path.join(self.base_dir,
                self.df.iloc[data_index, 0])

    img = mpimg.imread()

    ## this is where you preprocess the image
    ## make sure to resize it to be self.output_size

    label = self.df.iloc[data_index, 1:].to_numpy()
    ## if you have any preprocessing for
    ## the labels too do it here

    X[i,] = img
    y[i] = label

  return X, y


```

---

Now you are ready to fit the model to this generator. You can also easily make a validation generator and validate your model against that, all you need to do is make a new instance of the `DataGenerator` class, and pass in the validation csv and base directory and you're good to go. That's why I love OOP.

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
#  note you could also make a validation generator and pass it here like normal datasets

# back in the days you had to do this
# model.fit_generator(train_gen, ...)


```

---

## The complete code

```python

class DataGenerator(Sequence):

  def __init__(self, csv_file, base_dir, output_size, shuffle=False, batch_size=10):
    """
    Initializes a data generator object
      :param csv_file: file in which image names and numeric labels are stored
      :param base_dir: the directory in which all images are stored
      :param output_size: image output size after preprocessing
      :param shuffle: shuffle the data after each epoch
      :param batch_size: The size of each batch returned by __getitem__
    """
    self.df = pd.read_csv(csv_file)
    self.base_dir = base_dir
    self.output_size = output_size
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.on_epoch_end()

  def on_epoch_end(self):
    self.indices = np.arange(len(self.df))
    if self.shuffle:
      np.random.shuffle(self.indices)

  def __len__(self):
    return int(len(self.df) / self.batch_size)

  def __getitem__(self, idx):
    ## Initializing Batch
    #  that one in the shape is just for a one channel images
    # if you want to use colored images you might want to set that to 3
    X = np.empty((self.batch_size, *self.output_size, 1))
    # (x, y, h, w)
    y = np.empty((self.batch_size, 4, 1))

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

    for i, data_index in enumerate(indices):
      img_path = os.path.join(self.base_dir,
                  self.df.iloc[data_index, 0])

      img = mpimg.imread()
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # to reduce it to one channel to match the shape
      ## this is where you preprocess the image
      ## make sure to resize it to be self.output_size

      label = self.df.iloc[data_index, 1:].to_numpy()
      ## if you have any preprocessing for
      ## the labels too do it here

      X[i,] = img
      y[i] = label

    return X, y


## Defining and training the model

model = Sequential([
  ## define the model's architecture
])

train_gen = DataGenerator("data.csv", "data", (244, 244), batch_size=20, shuffle=True)

## compile the model first of course

# now let's train the model
model.fit(train_gen, epochs=5, ...)


```

---

And that's it.. you've just created your dataset generator that loads the data into memory batch by batch instead of the whole thing at once.  
I hope this was useful.

---

<a class="mdlink" href="https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly">Reference Article</a>

<a class="mdlink" href="https://github.com/mahmoudyusof/facial_keypoint_detection">My Notebooks</a>
