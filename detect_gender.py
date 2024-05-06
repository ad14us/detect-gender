import io, os, glob, shutil
import cv2
from PIL import Image
import streamlit as st


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import time
import uuid

from mtcnn import MTCNN
detector = MTCNN()

import subprocess; print(subprocess.run(['ls -la'], shell=True))
                                         
#@st.experimental_singleton
def load_model():
    model = tf.keras.models.load_model("./model_finishvgg16_30.h5")
    return model

model = load_model()

def load_test_image():
    

    uploaded_file = st.file_uploader(label='Upload file image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_test_image2():
    

    uploaded_file = st.file_uploader(label='Upload file image')
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image)
        image = load_img(uploaded_file, target_size=(218, 178))
        #image = image.resize((218, 178), Image.NEAREST)
        #preprocess the image
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        return image

def prepare_image():
    random_name = str(uuid.uuid4())
    working_dir = f"work_{random_name}"
    input_dir = f"{working_dir}/input"
    cropped_dir = f"{working_dir}/crop"
    os.mkdir(working_dir)
    os.mkdir(input_dir)
    os.mkdir(cropped_dir)

    return working_dir, input_dir

def load_test_image3():
    uploaded_file = st.file_uploader(label='Upload file image')
    st.write("Atau...")
    option_file = st.selectbox(
        "Pilih sample image berikut :",
        ("Person_1", "Person_2", "Group_People"),
        index=None,
        placeholder="Select one image...",
    )

    working_dir = None
    if 'working_dir' in st.session_state:
        working_dir = st.session_state['working_dir']

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        if 'working_dir' not in st.session_state:
            working_dir, input_dir = prepare_image()
            st.session_state['working_dir'] = working_dir
            with open(os.path.join(input_dir,"input_image.jpg"),"wb") as f: 
                f.write(uploaded_file.getbuffer())

        return working_dir
    elif option_file is not None:
        file_nm = f"{option_file}.jpg"
        image = Image.open(file_nm)
        st.image(image)
        if 'working_dir' not in st.session_state:
            working_dir, input_dir = prepare_image()
            st.session_state['working_dir'] = working_dir
            shutil.copyfile(file_nm, f"{input_dir}/input_image.jpg")

        return working_dir


def crop_faces_mtcnn(working_dir, scale_factor=1.8):
    
    path = f"{working_dir}/input/input_image.jpg"
    path_crop = f"{working_dir}/crop"

    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
    detections = detector.detect_faces(image)
    min_conf = 0.9
    cropped_files = []
    for i, det in enumerate(detections):
        if det['confidence'] >= min_conf:
            x, y, w, h = det['box']

            scaled_w = int(w * scale_factor)
            scaled_h = int(h * scale_factor * 0.8)
            scaled_x = max(0, x - (scaled_w - w) // 2)
            scaled_y = max(0, y - (scaled_h - h) // 2)
            scaled_w = min(scaled_w, image.shape[1] - scaled_x)
            scaled_h = min(scaled_h, image.shape[0] - scaled_y)

            cropped_face = image[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]
            cropped = f"cf{i}_cropped.jpg"
            cv2.imwrite(f"{path_crop}/{cropped}", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            cropped_files.append(cropped)
    return path_crop

def predict_face(working_dir):
    print(f"inside predict : {working_dir}")
    cropped_dir = os.path.join(working_dir, "crop")
    img_fn = sorted([ os.path.basename(x) for x in glob.glob(f"{cropped_dir}/*.jpg") ])
    img_dict = { 'Filenames' : img_fn }
    df_img = pd.DataFrame(img_dict)
    df_img['Path'] = f"{cropped_dir}/" + df_img['Filenames']
    IMG_SIZE = (218, 178)
    BATCH_SIZE = 1
    EPOCH_SIZE = 1

    other_gen = ImageDataGenerator(rescale=1./255)
    other_generator = other_gen.flow_from_dataframe(
        df_img, 
        cropped_dir,  
        x_col='Filenames',
        y_col=None,
        class_mode=None,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    predict = model.predict(other_generator)
    prediction = predict.argmax(axis=-1)
    i = 0
    for img in range(len(df_img))  :
        print(f"{img} : {prediction[i]}")
        i = i+1
    gender = ["Male" if ele == 1 else "Female" for ele in prediction]
    pathlist = df_img['Path'].to_list()
    st.image(pathlist,width=70, caption=gender)

    del st.session_state['working_dir']

    curr_dir = os.getcwd()
    del_dir = os.path.join(curr_dir,working_dir)

    shutil.rmtree(del_dir)
def main():
    st.title('Detect Gender')
    #st.subheader('Sebuah program sederhana yang memanfaatkan Deep Learning dengan arsitektur VGG-16. Program ini akan mendeteksi apakah itu wajah laki-laki atau perempuan')
    st.write('Sebuah program sederhana yang memanfaatkan Deep Learning dengan arsitektur VGG-16. Program ini akan mendeteksi apakah itu wajah laki-laki atau perempuan')
    st.write('Cara Pakai :  \n1. Upload sebuah file image atau gunakan sample image yang sudah tersedia.  \n2. Klik tombol Predict Gender di bawah')

    file_path = os.getcwd()

    working_dir = load_test_image3() 
    predictions = st.button('Predict Gender')
    if predictions:
        st.write('Prediction inprogress.. please wait')
        cropped_dir = crop_faces_mtcnn(working_dir)

        predict_face(working_dir)


if __name__ == '__main__':
    main()