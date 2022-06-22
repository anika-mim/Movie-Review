(%tensorflow_version 2.X
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

!pip install tensorflow --upgrade
 tfds.list_builders()
(train_data), test_data = tfds.load(name='imdb_reviews', split = (tfds.Split.TRAIN,tfds.Split.TEST),as_supervised=True)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(20)))
train_examples_batch
train_labels_batch

hub_layer(train_examples_batch[:2])
train_examples_batch[:5]

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data.shuffle(10000).batch(512),
         epochs=20,
         verbose=1)
value = input("Please enter a string:\n")
x=value
print ("The input is: \n")
print(x)
z=model.predict([x])
print(z)
pip install google_trans_new
from google_trans_new import google_translator  

translator = google_translator()  
x=input("Please write a review of the movie in any language: \n")
translate_text = translator.translate(x,lang_tgt='en')  
x=translate_text
y=model.predict([x])
if(y<0.35):
  print("Negative review")
elif(y>0.35 and y<0.65):
    print("Neutral review")
else:
  print("Positive review")


