import argparse 
import tensorflow as tf 
import tensorflow_hub as hub 
import numpy as np 
from PIL import Image 
import json 



# defining arguments for parser object:
pars_app = argparse.ArgumentParser()
pars_app.add_argument('--user_img', default='./test_images/cautleya_spicata.jpg')
pars_app.add_argument('--my_model',default='bestt_model.h5')
pars_app.add_argument('--topk',default=5)
pars_app.add_argument('--classes', default='label_map.json')

# values of the arguments
arguments = pars_app.parse_args()

image = arguments.user_img
my_model = arguments.my_model
TK = arguments.topk
label = arguments.classes

#reading json file and loading the model:
with open(label, 'r')as f:
    class_names = json.load(f)
    
model = tf.keras.models.load_model(my_model,
                                         custom_objects={'KerasLayer':hub.KerasLayer})

# defining functions for predict :

# 1- to re-shape the image to need the valid model shape of images:

def process_image(image):
    user_image = tf.convert_to_tensor(image)
    user_image = tf.image.resize(user_image,(224,224))
    user_image /= 255
    user_image = user_image.numpy()#.squeeze()
    return user_image

# 2- make the prediction :

def predict(image ,model,TK ):
        
    im = Image.open(image)
    to_pred_image = np.asarray(im)
    to_pred_image = process_image(to_pred_image)
    to_pred_image = np.expand_dims(to_pred_image,axis=0)
    
    ps = model.predict(to_pred_image)
    ps = ps.tolist()

    # to get the topK values and classes:
    values , indices = tf.math.top_k(ps,k=TK)
    probs = values.numpy().tolist()[0]
    classes = indices.numpy().tolist()[0]
    #convert classes integer labels to actual flower names :
    labels = [class_names[str(clas+1)]for clas in classes]
    print('the probabilites for there is image are:', probs)
    print('\n the classes for each prob. are :',labels)
    
    #return probs , classes



if __name__ == '__main__':
    predict(image ,model,TK)


