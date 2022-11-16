import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet169
from scipy.special import logit
from tf_keras_vis.scorecam import Scorecam
from matplotlib import cm
import time 


@st.cache(allow_output_mutation=True)
def adapt_DenseNet169() -> Model:
    """
    This code uses adapts the InceptionV3 with ImageNet weights to a regression
    problem. 

    Returns
    -------
    Model
        The keras model.
    """
    inputs = layers.Input(shape=(224, 224, 3))
    model = DenseNet169(include_top=False, input_tensor=inputs, weights='imagenet')
    # model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1000, activation = 'relu',name="dense_1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, activation = 'relu',name="dense_2")(x)
    x = layers.Dropout(0.5,name="top_dropout")(x)
    x = layers.Dense(100, activation = 'relu',name="dense_3")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, name="pred", activation='sigmoid')(x)

    model = Model(inputs,outputs,name="DenseNet169_over")
#     model.load_weights('./DenseNet169_over_sig.h5')

    return model

model = adapt_DenseNet169()
scorecam = Scorecam(model)

def score_function(output):
    return output[:,0]


logo = Image.open('tb-logo-lg.png')
st.sidebar.image(logo,width=200)
number_images = st.sidebar.selectbox('How many thumbnails do you want to compare?',(2,3,4,5,6,7,8,9,10))

left_column, right_column = st.columns(2)

tensor_images = []
images = []
ctr = []
uploaded_files = []
for i in range(number_images):
    uploaded_files.append(st.sidebar.file_uploader("Choose a thumbnail",type=['jpg','png','jpeg'],key=i))
    if uploaded_files[i]:
        image = Image.open(uploaded_files[i])
        left_column.image(image,caption='Thumbnail_' + str(i +1),width=250)
        image = np.array(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        height = image.shape[0]
        width = image.shape[1]
        image = cv2.resize(image,[224,224])/255 
        images.append(image)
        tensor_images.append(np.expand_dims(image, axis=0))

if st.sidebar.button('Analyze'):
    # start = time.time()
    for i in range(number_images):
        if uploaded_files[i]:
            pred = model.predict(tensor_images[i])[0][0]
            ctr.append(abs(logit(pred)))
            cam = scorecam(score_function, tensor_images[i], max_N=10)
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            result = cv2.addWeighted(images[i],0.5,heatmap/255,0.5,0)
            result = cv2.resize(result,[width,height])
            right_column.image(result,caption='Heat_map_' + str(i + 1),width=250,clamp=True)

    sorted_index = np.argsort(ctr)
    if uploaded_files[number_images - 1]:
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Thumbnail ' + str(sorted_index[number_images - 1] + 1) + ' wins</p>'
        left_column.write(new_title,unsafe_allow_html=True)
        left_column.write('**From the highest rated to the lowest:**')
        for i in range(number_images-1,-1,-1):
            left_column.write('Thumbnail ' + str(sorted_index[i] + 1), key=i)
            for j in range(i-1,-1,-1):
                increase_percent = 100 * (ctr[sorted_index[i]]-ctr[sorted_index[j]])/abs(ctr[sorted_index[j]])
                right_column.write("Increase percent between Thumbnail " + str(sorted_index[i] + 1) 
                + " and Thumbnail " + str(sorted_index[j] + 1) + " equals " + str(round(increase_percent,3)) + "%")
    # end = time.time()
    # st.write(end - start)
