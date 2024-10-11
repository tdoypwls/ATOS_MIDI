import os.path
import cv2 # pip3 install opencv-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from IPython.display import Markdown, display

# GPU Setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1은 오류 발생 0이 적당함

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# tensorflow GPU cuda 오류 시 아래 링크 설치 필요
# https://teddylee777.github.io/colab/tensorflow-gpu-install-windows


psd_df = pd.read_pickle('./dataframe/program_set_df.pkl')
isNewModelTrain = bool(psd_df['icfm_isNewModelTrain'].values[0])
isDFPC_DataSet = bool(psd_df['icfm_isDFPC_DataSet'].values[0])
isDFPC_users =  bool(psd_df['icfm_isDFPC_users'].values[0])
isDFPC_composer = bool(psd_df['icfm_isDFPC_composer'].values[0])
isDFPC_genre = bool(psd_df['icfm_isDFPC_genre'].values[0])
isDFPC_mood = bool(psd_df['icfm_isDFPC_mood'].values[0])
isDFPC_rhythm = bool(psd_df['icfm_isDFPC_rhythm'].values[0])
isDFPC_tag = bool(psd_df['icfm_isDFPC_tag'].values[0])
isDFPC_tempo = bool(psd_df['icfm_isDFPC_tempo'].values[0])
isWindows = bool(psd_df['isWindows'].values[0])


#isNewModelTrain = False
#isDFPC_users =  True
#isDFPC_composer = False
#isDFPC_DataSet = False
#isDFPC_rhythm = False
#isWindows = True

dir_str = './DFPC_mood/'
dir_ = Path(dir_str)
filepaths = list(dir_.glob(r'**/*.png'))

# 윈도우만 아래 코드 실행 (경로 '/'문자오류 보정)============
if isWindows:
    filepaths_str = []
    for i in filepaths:
        filepaths_str.append(str(i).replace('\\','/'))
    filepaths = filepaths_str

#이미지데이터의 경로와 label데이터로 데이터프레임 만들기 
def proc_img(filepath):
    
    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # 경로와 라벨 concatenate
    df = pd.concat([filepath, labels], axis=1)

    # index 재설정
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    
    return df

def create_gen():
    # 생성기 및 데이터 증강으로 이미지 로드
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    return train_generator,test_generator,train_images,val_images,test_images

def get_model(model):
# Load the pretained model
    kwargs =    {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
    
    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False # 레이어를 동결 시켜서 훈련중 손실을 최소화 한다.
    
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # 라벨 개수가 8개이기 때문에 Dencs도 8로 설정
    #outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    outputs = tf.keras.layers.Dense(labels_num, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# from PIL import Image
def printmd(string):
    # Print with Markdowns    
    display(Markdown(string))

def user_png_sim(path):
    class_dictionary = (train_images.class_indices)
    class_dictionary = dict((k,v) for k,v in class_dictionary.items())
    #print(class_dictionary)

    IMAGE_SIZE    = (224, 224)

    user_test_df = path
    test_image = image.load_img(user_test_df
                                ,target_size =IMAGE_SIZE )
    test_image = image.img_to_array(test_image)
    #plt.imshow(test_image/255.);

    test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    test_image = preprocess_input(test_image)
    prediction = model.predict(test_image)

    df = pd.DataFrame({'pred':prediction[0]})
    df = df.sort_values(by='pred', ascending=False, na_position='first')
    #printmd(f"## 예측률 : {(df.iloc[0]['pred'])* 100:.2f}%")
    #print(df)

    for x in class_dictionary:
      if class_dictionary[x] == (df[df == df.iloc[0]].index[0]):
        #printmd(f"### Class prediction = {x}")
        return x
        break    

# png 묶음(1곡)별 예측
def proc_test_img(filepath):
    """
   		이미지데이터의 경로와 label데이터로 데이터프레임 만들기 
    """

    labels = []

    for i in range(len(filepath)):
        if 'MID' in str(filepath[i]).split("/")[-1]:
            labels.append(str(filepath[i]).split("/")[-1].split("MID")[-2] + 'MID')
        else:
            labels.append(str(filepath[i]).split("/")[-1].split("mid")[-2] + 'mid')


    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # 경로와 라벨 concatenate
    df = pd.concat([filepath, labels], axis=1)

    # index 재설정
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    
    return df

def composer_cnn_df(path):

    dir_test = Path(path)
    filepaths_test = list(dir_test.glob(r'**/*.png'))

    #윈도우는 아래 작업 필요.
    if isWindows:
        filepaths_test_str = []
        for i in filepaths_test:
            filepaths_test_str.append(str(i).replace('\\','/'))
        filepaths_test = filepaths_test_str

    df_test = proc_test_img(filepaths_test)
    df_items = df_test['Label'].values.tolist()

    labelNames = [] # 중복 제거된 값들이 들어갈 리스트
    for value in df_items:
        if value not in labelNames:
            labelNames.append(value)

    #df_test_Mid # labelNames을 기준으로 특정 1곡에 대한 PNG 파일 묶음 데이터 프레임
    #df_test # 폴더 내부에 있는 모든 MIDI파일의 path / label 데이터 프레임


    df_list = []
    df_com = pd.DataFrame({
                      'Midi_name' : [],
                      'mood_cnn_best' : [],
                      'mood_cnn_best_score' : [],
                      'mood_cnn_sim' : []
                      })

    for labelName in labelNames:
        is_midiLabel = df_test['Label'] == labelName
        df_test_Mid = df_test[is_midiLabel]
        df_list.append(df_test_Mid)

    result_list = []

    for i in df_list:
        fname = i['Label'].values.tolist()[0]
        names = []
        for k in i['Filepath'].values.tolist():
            names.append(user_png_sim(k)) 
            
        count={}
        for j in names:
            try: count[j] += 1
            except: count[j]=1

        count_per = {}
        for key, value in count.items():
            count_per[key] = round(value/len(names),2)
            
        result = max(count, key=count.get)
        print(fname ,' best :',result, round(count[result]/len(names),2) ,count_per)

        df_append = [fname, result, round(count[result]/len(names),2),count_per]
        df_com = df_com.append(pd.Series(df_append, index=df_com.columns), ignore_index=True)


    return df_com


df = proc_img(filepaths)
print(df.head())

print(f'Number of pictures: {df.shape[0]}\n')
print(f'Number of different labels: {len(df.Label.unique())}\n')
print(f'Labels: {df.Label.unique()}')

labels_num = len(df.Label.unique())
labels_dic = df.Label.unique()



# DataSet check
fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.Filepath[i]))
    ax.set_title(df.Label[i], fontsize = 12)
plt.tight_layout(pad=0.5)
#plt.show()


# Number of pictures of each category
vc = df['Label'].value_counts()
plt.figure(figsize=(9,5))
sns.barplot(x = vc.index, y = vc, palette = "rocket")
plt.title("Number of pictures of each category", fontsize = 15)
#plt.show()


# Training/test split
# train_df,test_df = train_test_split(df.sample(frac=0.2), test_size=0.1,random_state=0) #모델링 시간이 오래걸리면 사용
train_df,test_df = train_test_split(df, test_size=0.1,random_state=0)
train_df.shape,test_df.shape
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   validation_split=0.2)
train_gen = train_datagen.flow_from_directory(dir_str,
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',subset='training')
val_gen  = train_datagen.flow_from_directory(dir_str,
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',subset='validation')
train_generator,test_generator,train_images,val_images,test_images=create_gen()
print('\n')


models = {
    "DenseNet121": {"model":tf.keras.applications.DenseNet121, "perf":0},
    "ResNet152V2": {"model":tf.keras.applications.ResNet152V2, "perf":0},
}

# Create the generators
train_generator,test_generator,train_images,val_images,test_images=create_gen()
print('\n')






path_model = './Models/'

if isNewModelTrain:
    # 모델 학습하기 (모델 불러오기 시 생략)
    train_df,test_df = train_test_split(df, test_size=0.1, random_state=0)
    train_generator,test_generator,train_images,val_images,test_images=create_gen()
    model = get_model(tf.keras.applications.ResNet152V2)
    history = model.fit(train_images,validation_data=val_images,epochs=25)
    model.save(path_model + 'img_cnn_for_mood_model.h5')

else:
    # 모델 불러오기 (새롭게 학습 시 생략 가능)
    model = load_model(path_model + 'img_cnn_for_mood_model.h5')




pkl_path = './dataframe/'

# DFPC_폴더 분석
if isDFPC_DataSet:
    dir_test_str = './DFPC_DataSet'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_DataSet_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_DataSet_mood_df.pkl')
    print(show_df)

if isDFPC_users:
    dir_test_str = './DFPC_users'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_users_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_users_mood_df.pkl')
    print(show_df)

if isDFPC_composer:
    dir_test_str = './DFPC_composer'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_composer_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_composer_mood_df.pkl')
    print(show_df)

if isDFPC_genre:
    dir_test_str = './DFPC_genre'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_genre_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_genre_mood_df.pkl')
    print(show_df)

if isDFPC_mood:
    dir_test_str = './DFPC_mood'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_mood_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_mood_mood_df.pkl')
    print(show_df)

if isDFPC_rhythm:
    dir_test_str = './DFPC_rhythm'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_rhythm_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_rhythm_mood_df.pkl')
    print(show_df)

if isDFPC_tag:
    dir_test_str = './DFPC_tag'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_tag_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_tag_mood_df.pkl')
    print(show_df)

if isDFPC_tempo:
    dir_test_str = './DFPC_tempo'
    save_pkl = composer_cnn_df(dir_test_str).to_pickle(pkl_path + 'DFPC_tempo_mood_df.pkl')
    show_df = pd.read_pickle(pkl_path + 'DFPC_tempo_mood_df.pkl')
    print(show_df)


"""
# current_user_song 만 분석할 때는 필요 없음.(pred변수 필요없음, 시간 오래걸림)
# Predict the label of the test_images
print('Predict the label of the test images')
pred = model.predict(test_images)
print('predict complet')
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

y_test = list(test_df.Label)
acc = accuracy_score(y_test,pred)
printmd(f'# Accuracy on the test set: {acc * 100:.2f}%')

class_report = classification_report(y_test, pred, zero_division=1)
print(class_report)

# Normalized Confusion Matrix
cf_matrix = confusion_matrix(y_test, pred, normalize='true')
plt.figure(figsize = (10,7))
sns.heatmap(cf_matrix, annot=False, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.show()

# current_user_song 만 분석할 때는 필요 없음.(pred변수 필요없음, 시간 오래걸림
# Display picture of the dataset with their labels
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 16),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i].split('_')[0]}\nPredicted: {pred[i].split('_')[0]}", fontsize = 15)
plt.tight_layout()
#plt.show()
"""
