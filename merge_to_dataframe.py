# MIDI file list to csv
from pathlib import Path
import pandas as pd
import openpyxl #엑셀 읽기쓰기모듈, 윈도우는 필요
import numpy as np



psd_df = pd.read_pickle('./dataframe/program_set_df.pkl')
isWindows = bool(psd_df['isWindows'].values[0])
search_song = str(psd_df['cont_name'].values[0])



# DataSet Dataframe
dir_str = './MIDIs/DataSet_MIDI'
dir_ = Path(dir_str)
filepaths = list(dir_.glob(r'**/*.MID'))

#윈도우는 아래 코드로 대체
if isWindows:
    filepaths_str = []
    for i in filepaths:
        filepaths_str.append(str(i).replace('\\','/'))
    filepaths = filepaths_str
else:
    filepaths2 = list(dir_.glob(r'**/*.mid'))
    filepaths = filepaths + filepaths2


composer_dir_str = './MIDIs/MDS_Composer'
composer_dir_ = Path(composer_dir_str)
composer_filepaths = list(composer_dir_.glob(r'**/*.MID'))

#윈도우는 아래 코드로 대체
if isWindows:
    composer_filepaths_str = []
    for i in composer_filepaths:
        composer_filepaths_str.append(str(i).replace('\\','/'))
    composer_filepaths = composer_filepaths_str
else:
    composer_filepaths2 = list(composer_dir_.glob(r'**/*.mid'))
    composer_filepaths = composer_filepaths + composer_filepaths2


def make_df(filepath):
    concat_list = []
    Midi_name = [str(filepath[i]).split("/")[-1] \
        for i in range(len(filepath))]
    Midi_name = pd.Series(Midi_name, name='Midi_name')
    concat_list.append(Midi_name)
    Genre_name = [str(filepath[i]).split("/")[-2] \
        for i in range(len(filepath))]
    Genre_name = pd.Series(Genre_name, name='Genre_name')
    concat_list.append(Genre_name)
    filepath = pd.Series(filepath, name='Filepath').astype(str)
    concat_list.append(filepath)
    
    # 경로와 라벨 concatenate
    df = pd.concat(concat_list, axis=1)

    # index 재설정
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    return df

df = make_df(filepaths)
#print(df)

def make_composer_df(filepath):
    concat_list = []
    Midi_name = [str(filepath[i]).split("/")[-1] \
        for i in range(len(filepath))]
    Midi_name = pd.Series(Midi_name, name='Midi_name')
    concat_list.append(Midi_name)

    Composer = [str(filepath[i]).split("/")[-2] \
        for i in range(len(filepath))]
    Composer = pd.Series(Composer, name='Composer')
    concat_list.append(Composer)

    #filepath = pd.Series(filepath, name='Filepath').astype(str)
    #concat_list.append(filepath)
    
    # 경로와 라벨 concatenate
    df = pd.concat(concat_list, axis=1)

    # index 재설정
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    return df

com_df = make_composer_df(composer_filepaths)
#print(com_df)
# merge df (genre + composer)

merge_df = pd.merge(df,com_df, how='left', on='Midi_name')
merge_df.to_excel('./dataframe/df.xlsx')
#print(merge_df)





#======================================================================================


# harmonic_reduction_similarity.py
# MIDI분석 및 화성진행 유사도 측정

path = './dataframe/'
harmonic_merge_df = pd.read_pickle(path + 'harmonic_reduction_merge_df.pkl')
harmonic_sim_df = pd.read_pickle(path + 'harmonic_reduction_sim_df.pkl')

harmonic_df = pd.merge(harmonic_merge_df,harmonic_sim_df, how='left', on='Midi_name')
harmonic_df = harmonic_df.sort_values('harmonic_reduction_score', ascending=False)

# hamoric_merge_df 와 harmonic_sim_df 데이터수 체크. 안맞으면 오류남
#print('harmonic_merge_df_count')
#print(harmonic_merge_df.count())
#print('harmonic_sim_df_count')
#print(harmonic_sim_df.count())


#======================================================================================


# key_signature로 major / minor 구분 및 점수 열 추가

user_filename = 'current_user_song.MID'
key_signature_ALL = psd_df['key_signature_ALL'].values[0]
#key_signature_ALL =True # True일경우 조성 전체가 일치해야 가산점 부여, False는 major minora

def key_sim_score(key):
    
    if key_signature_ALL:
        user_key_type = str(harmonic_df.loc[harmonic_df.Midi_name == user_filename].iloc[0]['key_signature']) # 유저곡 파일명을 직접 입력
        user_key_type = str(harmonic_df.iloc[0]['key_signature']) # harmonic_sim값이 가장 높은 것을 유저곡 기준으로 설정

    else:
        user_key_type = str(harmonic_df.loc[harmonic_df.Midi_name == user_filename].iloc[0]['key_signature']).split(' ')[-1] # 유저곡 파일명을 직접 입력
        user_key_type = str(harmonic_df.iloc[0]['key_signature']).split(' ')[-1] # harmonic_sim값이 가장 높은 것을 유저곡 기준으로 설정

    
    if key_signature_ALL:
        dataSet_key_type = str(key)
        
    else:
        dataSet_key_type = str(key).split(' ')[-1]
    

    if user_key_type == dataSet_key_type:
        result = 0.3  # 기준 곡과 조성(major or minor)가 일치하면 0.3점
    else:
        result = 0    # 기준 곡과 조성(major or minor)가 불일치하면 0점

    return result

harmonic_df['key_sim_score'] = harmonic_df['key_signature'].apply(key_sim_score)
print(harmonic_df)


#=============================== img_cnn_for_composer.py===============================
#======================================================================================
#======================================================================================

isWindows = True # 이미 선언한 변수라 생략 가능 편의상 기입
user_filename = 'current_user_song.MID'
df_path = './dataframe/'
folderpath = './MIDIs/MDS_Composer/'
user_pkl = 'DFPC_users_composer_df.pkl'
dataSet_pkl = 'DFPC_DataSet_composer_df.pkl'
sim_dict_column_name = 'Composer_cnn_sim' # 기존 데이터의 점수 dict 열 이름
score_column_name = 'Composer_cnn_score'  # df에 새로 추가되는 종합 점수 열 이름

# 가중치 세팅
correction = psd_df['composer_correction'].values[0]
correction_na = psd_df['composer_correction_na'].values[0]
correction_count = psd_df['composer_correction_count'].values[0]

#correction = 0.12
#correction_na = - 0.04
#correction_count = 0

# 작곡가별 유사도 측정
def make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name):

    # 폴더명을 라벨 리스트로 입력
    def get_label(folderpath):
        dir_str = folderpath
        dir_ = Path(dir_str)
        filepaths = list(dir_.glob(r'**/*.MID'))

        # 윈도우는 아래코드 필요
        if isWindows:
            filepaths_str = []
            for i in filepaths:
                filepaths_str.append(str(i).replace('\\','/'))
            filepaths = filepaths_str

        labels = [str(filepaths[i]).split("/")[-2] \
                for i in range(len(filepaths))]
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # 경로와 라벨 concatenate
        df = pd.concat([filepaths, labels], axis=1)
        # index 재설정
        df = df.sample(frac=1,random_state=0).reset_index(drop = True)
        df = df.Label.unique()
        return df
    Labels = get_label(folderpath)

    def make_label_dic(Labels, isUser):
        label_dic = {}
        for i in Labels:
            if isUser:
                label_dic[i] = i.lower() + '_us'
            else:
                label_dic[i] = i.lower() + '_s'
        return label_dic

    Labels_user_dic = make_label_dic( Labels, isUser=True )
    Labels_dic = make_label_dic( Labels, isUser=False )

    # df내부에서 apply 함수로 dic열을 분해
    def cnn_score(dic):
        for key, value in dic.items():
            if key == Labels[i]:
                return value
            else:
                None
        return 0

    # 기준곡 점수 부여
    def cnn_euclidean_score(dic):

        user_varName_dic = {}

        # user_score 변수지정
        for key, value in user_score_dic.items():
            for ikey, ivalue in Labels_user_dic.items():
                if key == ikey:
                    user_varName_dic[ivalue] = value

        # user_score에 포함되는 변수 리스트 확인
        user_var_list = []
        for key, value in user_score_dic.items():
            user_var_list.append(key)
        
        # user에 없는 변수가 있다면 그 변수를 생성하고 값을 0으로 설정
        for key, value in Labels_user_dic.items():
            if key not in user_var_list:
                user_varName_dic[value] = 0

        # dic 정렬
        user_varName_dic = dict(sorted(user_varName_dic.items()))
    #====
    #====
        dataSet_varName_dic = {}

        # DataSet_score 변수지정
        for key, value in dic.items():
            for ikey, ivalue in Labels_dic.items():
                if key == ikey:
                    dataSet_varName_dic[ivalue] = value

        dateSet_var_list = []
        for key, value in dic.items():
            dateSet_var_list.append(key)

        for key, value in Labels_dic.items():
            if key not in dateSet_var_list:
                dataSet_varName_dic[value] = 0

        # dic 정렬
        dataSet_varName_dic = dict(sorted(dataSet_varName_dic.items()))

        #print('userSetlist')
        #print(user_varName_dic)

        #print('dataSetlist')
        #print(dataSet_varName_dic)

        # Euclidean Distance 유클리디안 거리 계산 (n차원의 공간에서 두 점간의 거리 계싼)

        user_array_list = []
        for key, value in user_varName_dic.items():
            user_array_list.append(value)
        user_array_tuple = tuple(user_array_list)

        dataSet_array_list = []
        for key, value in dataSet_varName_dic.items():
            dataSet_array_list.append(value)
        dataSet_array_tuple = tuple(dataSet_array_list)   

        user_array = np.array(user_array_tuple)
        dataSet_array =  np.array(dataSet_array_tuple)

        result = - round(np.linalg.norm( user_array - dataSet_array ),2) * correction

        # Euclidean Distance 보정 (결측값이 0으로 표기되어 결측값에 대한 보정)
        # 보정튜닝이 잘 안돼서 코드지움

        return result


    #img_cnn_for_composer.py user_df 
    imgCnn_user_df = pd.read_pickle(df_path + user_pkl)
    imgCnn_user_df = imgCnn_user_df.loc[imgCnn_user_df.Midi_name == user_filename]

    for i in range(len(Labels)):
        imgCnn_user_df[Labels[i] + '_s'] = imgCnn_user_df[sim_dict_column_name].apply(cnn_score)

    # user_df는 current_user_song 그 자신으로 비교기준곡이라서 0점 부여 (가장 높은 점수)
    imgCnn_user_df[score_column_name] = 0
    user_score_dic = imgCnn_user_df[sim_dict_column_name].values[0]
    imgCnn_user_df = imgCnn_user_df.drop([ sim_dict_column_name ], axis=1)
    print(imgCnn_user_df)


    #img_cnn_for_composer.py DataSet_df 
    imgCnnDataSet_df = pd.read_pickle(df_path + dataSet_pkl)

    for i in range(len(Labels)):
        imgCnnDataSet_df[Labels[i] + '_s'] = imgCnnDataSet_df[sim_dict_column_name].apply(cnn_score)

    # 기준곡과 비교하여 작곡가 가산점 점수 계산 
    imgCnnDataSet_df[score_column_name] = imgCnnDataSet_df[ sim_dict_column_name ].apply(cnn_euclidean_score)
    imgCnnDataSet_df = imgCnnDataSet_df.drop([ sim_dict_column_name], axis=1)
    imgCnnDataSet_df = pd.concat([imgCnnDataSet_df, imgCnn_user_df])
    print(imgCnnDataSet_df)


    return imgCnnDataSet_df

imgCnnComposer_df = make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name)



#=============================== img_cnn_for_genre.py===============================
#======================================================================================
#======================================================================================

isWindows = True # 이미 선언한 변수라 생략 가능 편의상 기입
user_filename = 'current_user_song.MID'
df_path = './dataframe/'
folderpath = './MIDIs/MDS_Genre/'
user_pkl = 'DFPC_users_genre_df.pkl'
dataSet_pkl = 'DFPC_DataSet_genre_df.pkl'
sim_dict_column_name = 'Genre_cnn_sim' # 기존 데이터의 점수 dict 열 이름
score_column_name = 'Genre_cnn_score'  # df에 새로 추가되는 종합 점수 열 이름

# 가중치 세팅
correction = psd_df['genre_correction'].values[0]
correction_na = psd_df['genre_correction_na'].values[0]
correction_count = psd_df['genre_correction_count'].values[0]

#correction = 0.12
#correction_na = - 0.04
#correction_count = 0

# 작곡가별 유사도 측정
def make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name):

    # 폴더명을 라벨 리스트로 입력
    def get_label(folderpath):
        dir_str = folderpath
        dir_ = Path(dir_str)
        filepaths = list(dir_.glob(r'**/*.MID'))

        # 윈도우는 아래코드 필요
        if isWindows:
            filepaths_str = []
            for i in filepaths:
                filepaths_str.append(str(i).replace('\\','/'))
            filepaths = filepaths_str

        labels = [str(filepaths[i]).split("/")[-2] \
                for i in range(len(filepaths))]
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # 경로와 라벨 concatenate
        df = pd.concat([filepaths, labels], axis=1)
        # index 재설정
        df = df.sample(frac=1,random_state=0).reset_index(drop = True)
        df = df.Label.unique()
        return df
    Labels = get_label(folderpath)

    def make_label_dic(Labels, isUser):
        label_dic = {}
        for i in Labels:
            if isUser:
                label_dic[i] = i.lower() + '_us'
            else:
                label_dic[i] = i.lower() + '_s'
        return label_dic

    Labels_user_dic = make_label_dic( Labels, isUser=True )
    Labels_dic = make_label_dic( Labels, isUser=False )

    # df내부에서 apply 함수로 dic열을 분해
    def cnn_score(dic):
        for key, value in dic.items():
            if key == Labels[i]:
                return value
            else:
                None
        return 0

    # 기준곡 점수 부여
    def cnn_euclidean_score(dic):

        user_varName_dic = {}

        # user_score 변수지정
        for key, value in user_score_dic.items():
            for ikey, ivalue in Labels_user_dic.items():
                if key == ikey:
                    user_varName_dic[ivalue] = value

        # user_score에 포함되는 변수 리스트 확인
        user_var_list = []
        for key, value in user_score_dic.items():
            user_var_list.append(key)
        
        # user에 없는 변수가 있다면 그 변수를 생성하고 값을 0으로 설정
        for key, value in Labels_user_dic.items():
            if key not in user_var_list:
                user_varName_dic[value] = 0

        # dic 정렬
        user_varName_dic = dict(sorted(user_varName_dic.items()))
    #====
    #====
        dataSet_varName_dic = {}

        # DataSet_score 변수지정
        for key, value in dic.items():
            for ikey, ivalue in Labels_dic.items():
                if key == ikey:
                    dataSet_varName_dic[ivalue] = value

        dateSet_var_list = []
        for key, value in dic.items():
            dateSet_var_list.append(key)

        for key, value in Labels_dic.items():
            if key not in dateSet_var_list:
                dataSet_varName_dic[value] = 0

        # dic 정렬
        dataSet_varName_dic = dict(sorted(dataSet_varName_dic.items()))

        #print('userSetlist')
        #print(user_varName_dic)

        #print('dataSetlist')
        #print(dataSet_varName_dic)

        # Euclidean Distance 유클리디안 거리 계산 (n차원의 공간에서 두 점간의 거리 계싼)

        user_array_list = []
        for key, value in user_varName_dic.items():
            user_array_list.append(value)
        user_array_tuple = tuple(user_array_list)

        dataSet_array_list = []
        for key, value in dataSet_varName_dic.items():
            dataSet_array_list.append(value)
        dataSet_array_tuple = tuple(dataSet_array_list)   

        user_array = np.array(user_array_tuple)
        dataSet_array =  np.array(dataSet_array_tuple)

        result = - round(np.linalg.norm( user_array - dataSet_array ),2) * correction

        # Euclidean Distance 보정 (결측값이 0으로 표기되어 결측값에 대한 보정)
        # 보정튜닝이 잘 안돼서 코드지움

        return result


    #img_cnn_for_composer.py user_df 
    imgCnn_user_df = pd.read_pickle(df_path + user_pkl)
    imgCnn_user_df = imgCnn_user_df.loc[imgCnn_user_df.Midi_name == user_filename]

    for i in range(len(Labels)):
        imgCnn_user_df[Labels[i] + '_s'] = imgCnn_user_df[sim_dict_column_name].apply(cnn_score)

    # user_df는 current_user_song 그 자신으로 비교기준곡이라서 0점 부여 (가장 높은 점수)
    imgCnn_user_df[score_column_name] = 0
    user_score_dic = imgCnn_user_df[sim_dict_column_name].values[0]
    imgCnn_user_df = imgCnn_user_df.drop([ sim_dict_column_name ], axis=1)
    print(imgCnn_user_df)


    #img_cnn_for_composer.py DataSet_df 
    imgCnnDataSet_df = pd.read_pickle(df_path + dataSet_pkl)

    for i in range(len(Labels)):
        imgCnnDataSet_df[Labels[i] + '_s'] = imgCnnDataSet_df[sim_dict_column_name].apply(cnn_score)

    # 기준곡과 비교하여 작곡가 가산점 점수 계산 
    imgCnnDataSet_df[score_column_name] = imgCnnDataSet_df[ sim_dict_column_name ].apply(cnn_euclidean_score)
    imgCnnDataSet_df = imgCnnDataSet_df.drop([ sim_dict_column_name], axis=1)
    imgCnnDataSet_df = pd.concat([imgCnnDataSet_df, imgCnn_user_df])
    print(imgCnnDataSet_df)


    return imgCnnDataSet_df

imgCnnGenre_df = make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name)




#=============================== img_cnn_for_mood.py===============================
#======================================================================================
#======================================================================================

isWindows = True # 이미 선언한 변수라 생략 가능 편의상 기입
user_filename = 'current_user_song.MID'
df_path = './dataframe/'
folderpath = './MIDIs/MDS_Mood/'
user_pkl = 'DFPC_users_mood_df.pkl'
dataSet_pkl = 'DFPC_DataSet_mood_df.pkl'
sim_dict_column_name = 'mood_cnn_sim' # 기존 데이터의 점수 dict 열 이름
score_column_name = 'mood_cnn_score'  # df에 새로 추가되는 종합 점수 열 이름

# 가중치 세팅
correction = psd_df['mood_correction'].values[0]
correction_na = psd_df['mood_correction_na'].values[0]
correction_count = psd_df['mood_correction_count'].values[0]

#correction = 0.12
#correction_na = - 0.04
#correction_count = 0

# 작곡가별 유사도 측정
def make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name):

    # 폴더명을 라벨 리스트로 입력
    def get_label(folderpath):
        dir_str = folderpath
        dir_ = Path(dir_str)
        filepaths = list(dir_.glob(r'**/*.MID'))

        # 윈도우는 아래코드 필요
        if isWindows:
            filepaths_str = []
            for i in filepaths:
                filepaths_str.append(str(i).replace('\\','/'))
            filepaths = filepaths_str

        labels = [str(filepaths[i]).split("/")[-2] \
                for i in range(len(filepaths))]
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # 경로와 라벨 concatenate
        df = pd.concat([filepaths, labels], axis=1)
        # index 재설정
        df = df.sample(frac=1,random_state=0).reset_index(drop = True)
        df = df.Label.unique()
        return df
    Labels = get_label(folderpath)

    def make_label_dic(Labels, isUser):
        label_dic = {}
        for i in Labels:
            if isUser:
                label_dic[i] = i.lower() + '_us'
            else:
                label_dic[i] = i.lower() + '_s'
        return label_dic

    Labels_user_dic = make_label_dic( Labels, isUser=True )
    Labels_dic = make_label_dic( Labels, isUser=False )

    # df내부에서 apply 함수로 dic열을 분해
    def cnn_score(dic):
        for key, value in dic.items():
            if key == Labels[i]:
                return value
            else:
                None
        return 0

    # 기준곡 점수 부여
    def cnn_euclidean_score(dic):

        user_varName_dic = {}

        # user_score 변수지정
        for key, value in user_score_dic.items():
            for ikey, ivalue in Labels_user_dic.items():
                if key == ikey:
                    user_varName_dic[ivalue] = value

        # user_score에 포함되는 변수 리스트 확인
        user_var_list = []
        for key, value in user_score_dic.items():
            user_var_list.append(key)
        
        # user에 없는 변수가 있다면 그 변수를 생성하고 값을 0으로 설정
        for key, value in Labels_user_dic.items():
            if key not in user_var_list:
                user_varName_dic[value] = 0

        # dic 정렬
        user_varName_dic = dict(sorted(user_varName_dic.items()))
    #====
    #====
        dataSet_varName_dic = {}

        # DataSet_score 변수지정
        for key, value in dic.items():
            for ikey, ivalue in Labels_dic.items():
                if key == ikey:
                    dataSet_varName_dic[ivalue] = value

        dateSet_var_list = []
        for key, value in dic.items():
            dateSet_var_list.append(key)

        for key, value in Labels_dic.items():
            if key not in dateSet_var_list:
                dataSet_varName_dic[value] = 0

        # dic 정렬
        dataSet_varName_dic = dict(sorted(dataSet_varName_dic.items()))

        #print('userSetlist')
        #print(user_varName_dic)

        #print('dataSetlist')
        #print(dataSet_varName_dic)

        # Euclidean Distance 유클리디안 거리 계산 (n차원의 공간에서 두 점간의 거리 계싼)

        user_array_list = []
        for key, value in user_varName_dic.items():
            user_array_list.append(value)
        user_array_tuple = tuple(user_array_list)

        dataSet_array_list = []
        for key, value in dataSet_varName_dic.items():
            dataSet_array_list.append(value)
        dataSet_array_tuple = tuple(dataSet_array_list)   

        user_array = np.array(user_array_tuple)
        dataSet_array =  np.array(dataSet_array_tuple)

        result = - round(np.linalg.norm( user_array - dataSet_array ),2) * correction

        # Euclidean Distance 보정 (결측값이 0으로 표기되어 결측값에 대한 보정)
        # 보정튜닝이 잘 안돼서 코드지움

        return result


    #img_cnn_for_composer.py user_df 
    imgCnn_user_df = pd.read_pickle(df_path + user_pkl)
    imgCnn_user_df = imgCnn_user_df.loc[imgCnn_user_df.Midi_name == user_filename]

    for i in range(len(Labels)):
        imgCnn_user_df[Labels[i] + '_s'] = imgCnn_user_df[sim_dict_column_name].apply(cnn_score)

    # user_df는 current_user_song 그 자신으로 비교기준곡이라서 0점 부여 (가장 높은 점수)
    imgCnn_user_df[score_column_name] = 0
    user_score_dic = imgCnn_user_df[sim_dict_column_name].values[0]
    imgCnn_user_df = imgCnn_user_df.drop([ sim_dict_column_name ], axis=1)
    print(imgCnn_user_df)


    #img_cnn_for_composer.py DataSet_df 
    imgCnnDataSet_df = pd.read_pickle(df_path + dataSet_pkl)

    for i in range(len(Labels)):
        imgCnnDataSet_df[Labels[i] + '_s'] = imgCnnDataSet_df[sim_dict_column_name].apply(cnn_score)

    # 기준곡과 비교하여 작곡가 가산점 점수 계산 
    imgCnnDataSet_df[score_column_name] = imgCnnDataSet_df[ sim_dict_column_name ].apply(cnn_euclidean_score)
    imgCnnDataSet_df = imgCnnDataSet_df.drop([ sim_dict_column_name], axis=1)
    imgCnnDataSet_df = pd.concat([imgCnnDataSet_df, imgCnn_user_df])
    print(imgCnnDataSet_df)


    return imgCnnDataSet_df

imgCnnMood_df = make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name)





#=============================== img_cnn_for_rhythm.py=================================
#======================================================================================
#======================================================================================

folderpath = './MIDIs/MDS_Rhythm/'
user_pkl = 'DFPC_users_rhythm_df.pkl'
dataSet_pkl = 'DFPC_DataSet_rhythm_df.pkl'
sim_dict_column_name = 'Rhythm_cnn_sim' # 기존 데이터의 점수 dict 열 이름
score_column_name = 'Rhythm_cnn_score'  # df에 새로 추가되는 종합 점수 열 이름

# 가중치 세팅

correction = psd_df['rhythm_correction'].values[0]
correction_na = psd_df['rhythm_correction_na'].values[0]
correction_count = psd_df['rhythm_correction_count'].values[0]

#correction = 0.12
#correction_na = - 0.04
#correction_count = 0

# 리듬별 유사도 측정
def make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name):

    # 폴더명을 라벨 리스트로 입력
    def get_label(folderpath):
        dir_str = folderpath
        dir_ = Path(dir_str)
        filepaths = list(dir_.glob(r'**/*.MID'))

        # 윈도우는 아래코드 필요
        if isWindows:
            filepaths_str = []
            for i in filepaths:
                filepaths_str.append(str(i).replace('\\','/'))
            filepaths = filepaths_str

        labels = [str(filepaths[i]).split("/")[-2] \
                for i in range(len(filepaths))]
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # 경로와 라벨 concatenate
        df = pd.concat([filepaths, labels], axis=1)
        # index 재설정
        df = df.sample(frac=1,random_state=0).reset_index(drop = True)
        df = df.Label.unique()
        return df
    Labels = get_label(folderpath)

    def make_label_dic(Labels, isUser):
        label_dic = {}
        for i in Labels:
            if isUser:
                label_dic[i] = i.lower() + '_us'
            else:
                label_dic[i] = i.lower() + '_s'
        return label_dic

    Labels_user_dic = make_label_dic( Labels, isUser=True )
    Labels_dic = make_label_dic( Labels, isUser=False )

    # df내부에서 apply 함수로 dic열을 분해
    def cnn_score(dic):
        for key, value in dic.items():
            if key == Labels[i]:
                return value
            else:
                None
        return 0

    # 기준곡 점수 부여
    def cnn_euclidean_score(dic):

        user_varName_dic = {}

        # user_score 변수지정
        for key, value in user_score_dic.items():
            for ikey, ivalue in Labels_user_dic.items():
                if key == ikey:
                    user_varName_dic[ivalue] = value

        # user_score에 포함되는 변수 리스트 확인
        user_var_list = []
        for key, value in user_score_dic.items():
            user_var_list.append(key)
        
        # user에 없는 변수가 있다면 그 변수를 생성하고 값을 0으로 설정
        for key, value in Labels_user_dic.items():
            if key not in user_var_list:
                user_varName_dic[value] = 0

        # dic 정렬
        user_varName_dic = dict(sorted(user_varName_dic.items()))
    #====
    #====


        dataSet_varName_dic = {}

        # DataSet_score 변수지정
        for key, value in dic.items():
            for ikey, ivalue in Labels_dic.items():
                if key == ikey:
                    dataSet_varName_dic[ivalue] = value

        dateSet_var_list = []
        for key, value in dic.items():
            dateSet_var_list.append(key)

        for key, value in Labels_dic.items():
            if key not in dateSet_var_list:
                dataSet_varName_dic[value] = 0

        # dic 정렬
        dataSet_varName_dic = dict(sorted(dataSet_varName_dic.items()))

        #print('userSetlist')
        #print(user_varName_dic)

        #print('dataSetlist')
        #print(dataSet_varName_dic)

        # Euclidean Distance 유클리디안 거리 계산 (n차원의 공간에서 두 점간의 거리 계싼)

        user_array_list = []
        for key, value in user_varName_dic.items():
            user_array_list.append(value)
        user_array_tuple = tuple(user_array_list)

        dataSet_array_list = []
        for key, value in dataSet_varName_dic.items():
            dataSet_array_list.append(value)
        dataSet_array_tuple = tuple(dataSet_array_list)   

        user_array = np.array(user_array_tuple)
        dataSet_array =  np.array(dataSet_array_tuple)

        result = - round(np.linalg.norm( user_array - dataSet_array ),2) * correction

        # Euclidean Distance 보정 (결측값이 0으로 표기되어 결측값에 대한 보정)
        # 보정튜닝이 잘 안돼서 코드지움

        return result


    #img_cnn_for_composer.py user_df 
    imgCnn_user_df = pd.read_pickle(df_path + user_pkl)
    imgCnn_user_df = imgCnn_user_df.loc[imgCnn_user_df.Midi_name == user_filename]

    for i in range(len(Labels)):
        imgCnn_user_df[Labels[i] + '_s'] = imgCnn_user_df[sim_dict_column_name].apply(cnn_score)

    # user_df는 current_user_song 그 자신으로 비교기준곡이라서 0점 부여 (가장 높은 점수)
    imgCnn_user_df[score_column_name] = 0
    user_score_dic = imgCnn_user_df[sim_dict_column_name].values[0]
    imgCnn_user_df = imgCnn_user_df.drop([ sim_dict_column_name ], axis=1)
    print(imgCnn_user_df)


    #img_cnn_for_composer.py DataSet_df 
    imgCnnDataSet_df = pd.read_pickle(df_path + dataSet_pkl)

    for i in range(len(Labels)):
        imgCnnDataSet_df[Labels[i] + '_s'] = imgCnnDataSet_df[sim_dict_column_name].apply(cnn_score)

    # 기준곡과 비교하여 작곡가 가산점 점수 계산 
    imgCnnDataSet_df[score_column_name] = imgCnnDataSet_df[ sim_dict_column_name ].apply(cnn_euclidean_score)
    imgCnnDataSet_df = imgCnnDataSet_df.drop([ sim_dict_column_name], axis=1)
    imgCnnDataSet_df = pd.concat([imgCnnDataSet_df, imgCnn_user_df])
    print(imgCnnDataSet_df)


    return imgCnnDataSet_df

imgCnnRhythm_df = make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name)



#=============================== img_cnn_for_tag.py=================================
#======================================================================================
#======================================================================================

folderpath = './MIDIs/MDS_Tag/'
user_pkl = 'DFPC_users_tag_df.pkl'
dataSet_pkl = 'DFPC_DataSet_tag_df.pkl'
sim_dict_column_name = 'tag_cnn_sim' # 기존 데이터의 점수 dict 열 이름
score_column_name = 'tag_cnn_score'  # df에 새로 추가되는 종합 점수 열 이름

# 가중치 세팅

correction = psd_df['tag_correction'].values[0]
correction_na = psd_df['tag_correction_na'].values[0]
correction_count = psd_df['tag_correction_count'].values[0]

#correction = 0.12
#correction_na = - 0.04
#correction_count = 0

# 리듬별 유사도 측정
def make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name):

    # 폴더명을 라벨 리스트로 입력
    def get_label(folderpath):
        dir_str = folderpath
        dir_ = Path(dir_str)
        filepaths = list(dir_.glob(r'**/*.MID'))

        # 윈도우는 아래코드 필요
        if isWindows:
            filepaths_str = []
            for i in filepaths:
                filepaths_str.append(str(i).replace('\\','/'))
            filepaths = filepaths_str

        labels = [str(filepaths[i]).split("/")[-2] \
                for i in range(len(filepaths))]
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # 경로와 라벨 concatenate
        df = pd.concat([filepaths, labels], axis=1)
        # index 재설정
        df = df.sample(frac=1,random_state=0).reset_index(drop = True)
        df = df.Label.unique()
        return df
    Labels = get_label(folderpath)

    def make_label_dic(Labels, isUser):
        label_dic = {}
        for i in Labels:
            if isUser:
                label_dic[i] = i.lower() + '_us'
            else:
                label_dic[i] = i.lower() + '_s'
        return label_dic

    Labels_user_dic = make_label_dic( Labels, isUser=True )
    Labels_dic = make_label_dic( Labels, isUser=False )

    # df내부에서 apply 함수로 dic열을 분해
    def cnn_score(dic):
        for key, value in dic.items():
            if key == Labels[i]:
                return value
            else:
                None
        return 0

    # 기준곡 점수 부여
    def cnn_euclidean_score(dic):

        user_varName_dic = {}

        # user_score 변수지정
        for key, value in user_score_dic.items():
            for ikey, ivalue in Labels_user_dic.items():
                if key == ikey:
                    user_varName_dic[ivalue] = value

        # user_score에 포함되는 변수 리스트 확인
        user_var_list = []
        for key, value in user_score_dic.items():
            user_var_list.append(key)
        
        # user에 없는 변수가 있다면 그 변수를 생성하고 값을 0으로 설정
        for key, value in Labels_user_dic.items():
            if key not in user_var_list:
                user_varName_dic[value] = 0

        # dic 정렬
        user_varName_dic = dict(sorted(user_varName_dic.items()))
    #====
    #====


        dataSet_varName_dic = {}

        # DataSet_score 변수지정
        for key, value in dic.items():
            for ikey, ivalue in Labels_dic.items():
                if key == ikey:
                    dataSet_varName_dic[ivalue] = value

        dateSet_var_list = []
        for key, value in dic.items():
            dateSet_var_list.append(key)

        for key, value in Labels_dic.items():
            if key not in dateSet_var_list:
                dataSet_varName_dic[value] = 0

        # dic 정렬
        dataSet_varName_dic = dict(sorted(dataSet_varName_dic.items()))

        #print('userSetlist')
        #print(user_varName_dic)

        #print('dataSetlist')
        #print(dataSet_varName_dic)

        # Euclidean Distance 유클리디안 거리 계산 (n차원의 공간에서 두 점간의 거리 계싼)

        user_array_list = []
        for key, value in user_varName_dic.items():
            user_array_list.append(value)
        user_array_tuple = tuple(user_array_list)

        dataSet_array_list = []
        for key, value in dataSet_varName_dic.items():
            dataSet_array_list.append(value)
        dataSet_array_tuple = tuple(dataSet_array_list)   

        user_array = np.array(user_array_tuple)
        dataSet_array =  np.array(dataSet_array_tuple)

        result = - round(np.linalg.norm( user_array - dataSet_array ),2) * correction

        # Euclidean Distance 보정 (결측값이 0으로 표기되어 결측값에 대한 보정)
        # 보정튜닝이 잘 안돼서 코드지움

        return result


    #img_cnn_for_composer.py user_df 
    imgCnn_user_df = pd.read_pickle(df_path + user_pkl)
    imgCnn_user_df = imgCnn_user_df.loc[imgCnn_user_df.Midi_name == user_filename]

    for i in range(len(Labels)):
        imgCnn_user_df[Labels[i] + '_s'] = imgCnn_user_df[sim_dict_column_name].apply(cnn_score)

    # user_df는 current_user_song 그 자신으로 비교기준곡이라서 0점 부여 (가장 높은 점수)
    imgCnn_user_df[score_column_name] = 0
    user_score_dic = imgCnn_user_df[sim_dict_column_name].values[0]
    imgCnn_user_df = imgCnn_user_df.drop([ sim_dict_column_name ], axis=1)
    print(imgCnn_user_df)


    #img_cnn_for_composer.py DataSet_df 
    imgCnnDataSet_df = pd.read_pickle(df_path + dataSet_pkl)

    for i in range(len(Labels)):
        imgCnnDataSet_df[Labels[i] + '_s'] = imgCnnDataSet_df[sim_dict_column_name].apply(cnn_score)

    # 기준곡과 비교하여 작곡가 가산점 점수 계산 
    imgCnnDataSet_df[score_column_name] = imgCnnDataSet_df[ sim_dict_column_name ].apply(cnn_euclidean_score)
    imgCnnDataSet_df = imgCnnDataSet_df.drop([ sim_dict_column_name], axis=1)
    imgCnnDataSet_df = pd.concat([imgCnnDataSet_df, imgCnn_user_df])
    print(imgCnnDataSet_df)


    return imgCnnDataSet_df

imgCnnTag_df = make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name)




#=============================== img_cnn_for_tempo.py=================================
#======================================================================================
#======================================================================================

folderpath = './MIDIs/MDS_Tempo/'
user_pkl = 'DFPC_users_tempo_df.pkl'
dataSet_pkl = 'DFPC_DataSet_tempo_df.pkl'
sim_dict_column_name = 'tempo_cnn_sim' # 기존 데이터의 점수 dict 열 이름
score_column_name = 'tempo_cnn_score'  # df에 새로 추가되는 종합 점수 열 이름

# 가중치 세팅

correction = psd_df['tempo_correction'].values[0]
correction_na = psd_df['tempo_correction_na'].values[0]
correction_count = psd_df['tempo_correction_count'].values[0]

#correction = 0.12
#correction_na = - 0.04
#correction_count = 0

# 리듬별 유사도 측정
def make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name):

    # 폴더명을 라벨 리스트로 입력
    def get_label(folderpath):
        dir_str = folderpath
        dir_ = Path(dir_str)
        filepaths = list(dir_.glob(r'**/*.MID'))

        # 윈도우는 아래코드 필요
        if isWindows:
            filepaths_str = []
            for i in filepaths:
                filepaths_str.append(str(i).replace('\\','/'))
            filepaths = filepaths_str

        labels = [str(filepaths[i]).split("/")[-2] \
                for i in range(len(filepaths))]
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # 경로와 라벨 concatenate
        df = pd.concat([filepaths, labels], axis=1)
        # index 재설정
        df = df.sample(frac=1,random_state=0).reset_index(drop = True)
        df = df.Label.unique()
        return df
    Labels = get_label(folderpath)

    def make_label_dic(Labels, isUser):
        label_dic = {}
        for i in Labels:
            if isUser:
                label_dic[i] = i.lower() + '_us'
            else:
                label_dic[i] = i.lower() + '_s'
        return label_dic

    Labels_user_dic = make_label_dic( Labels, isUser=True )
    Labels_dic = make_label_dic( Labels, isUser=False )

    # df내부에서 apply 함수로 dic열을 분해
    def cnn_score(dic):
        for key, value in dic.items():
            if key == Labels[i]:
                return value
            else:
                None
        return 0

    # 기준곡 점수 부여
    def cnn_euclidean_score(dic):

        user_varName_dic = {}

        # user_score 변수지정
        for key, value in user_score_dic.items():
            for ikey, ivalue in Labels_user_dic.items():
                if key == ikey:
                    user_varName_dic[ivalue] = value

        # user_score에 포함되는 변수 리스트 확인
        user_var_list = []
        for key, value in user_score_dic.items():
            user_var_list.append(key)
        
        # user에 없는 변수가 있다면 그 변수를 생성하고 값을 0으로 설정
        for key, value in Labels_user_dic.items():
            if key not in user_var_list:
                user_varName_dic[value] = 0

        # dic 정렬
        user_varName_dic = dict(sorted(user_varName_dic.items()))
    #====
    #====


        dataSet_varName_dic = {}

        # DataSet_score 변수지정
        for key, value in dic.items():
            for ikey, ivalue in Labels_dic.items():
                if key == ikey:
                    dataSet_varName_dic[ivalue] = value

        dateSet_var_list = []
        for key, value in dic.items():
            dateSet_var_list.append(key)

        for key, value in Labels_dic.items():
            if key not in dateSet_var_list:
                dataSet_varName_dic[value] = 0

        # dic 정렬
        dataSet_varName_dic = dict(sorted(dataSet_varName_dic.items()))

        #print('userSetlist')
        #print(user_varName_dic)

        #print('dataSetlist')
        #print(dataSet_varName_dic)

        # Euclidean Distance 유클리디안 거리 계산 (n차원의 공간에서 두 점간의 거리 계싼)

        user_array_list = []
        for key, value in user_varName_dic.items():
            user_array_list.append(value)
        user_array_tuple = tuple(user_array_list)

        dataSet_array_list = []
        for key, value in dataSet_varName_dic.items():
            dataSet_array_list.append(value)
        dataSet_array_tuple = tuple(dataSet_array_list)   

        user_array = np.array(user_array_tuple)
        dataSet_array =  np.array(dataSet_array_tuple)

        result = - round(np.linalg.norm( user_array - dataSet_array ),2) * correction

        # Euclidean Distance 보정 (결측값이 0으로 표기되어 결측값에 대한 보정)
        # 보정튜닝이 잘 안돼서 코드지움

        return result


    #img_cnn_for_composer.py user_df 
    imgCnn_user_df = pd.read_pickle(df_path + user_pkl)
    imgCnn_user_df = imgCnn_user_df.loc[imgCnn_user_df.Midi_name == user_filename]

    for i in range(len(Labels)):
        imgCnn_user_df[Labels[i] + '_s'] = imgCnn_user_df[sim_dict_column_name].apply(cnn_score)

    # user_df는 current_user_song 그 자신으로 비교기준곡이라서 0점 부여 (가장 높은 점수)
    imgCnn_user_df[score_column_name] = 0
    user_score_dic = imgCnn_user_df[sim_dict_column_name].values[0]
    imgCnn_user_df = imgCnn_user_df.drop([ sim_dict_column_name ], axis=1)
    print(imgCnn_user_df)


    #img_cnn_for_composer.py DataSet_df 
    imgCnnDataSet_df = pd.read_pickle(df_path + dataSet_pkl)

    for i in range(len(Labels)):
        imgCnnDataSet_df[Labels[i] + '_s'] = imgCnnDataSet_df[sim_dict_column_name].apply(cnn_score)

    # 기준곡과 비교하여 작곡가 가산점 점수 계산 
    imgCnnDataSet_df[score_column_name] = imgCnnDataSet_df[ sim_dict_column_name ].apply(cnn_euclidean_score)
    imgCnnDataSet_df = imgCnnDataSet_df.drop([ sim_dict_column_name], axis=1)
    imgCnnDataSet_df = pd.concat([imgCnnDataSet_df, imgCnn_user_df])
    print(imgCnnDataSet_df)


    return imgCnnDataSet_df

imgCnnTempo_df = make_cnn_score_df(isWindows, df_path, folderpath, user_pkl, dataSet_pkl, user_filename, sim_dict_column_name, score_column_name)



#======================================================================================
# 종합 점수 계산


score_df = pd.merge(harmonic_df,imgCnnComposer_df, how='left', on='Midi_name')
score_df = pd.merge(score_df,imgCnnGenre_df, how='left', on='Midi_name')
score_df = pd.merge(score_df,imgCnnMood_df, how='left', on='Midi_name')
score_df = pd.merge(score_df,imgCnnRhythm_df, how='left', on='Midi_name')
score_df = pd.merge(score_df,imgCnnTag_df, how='left', on='Midi_name')
score_df = pd.merge(score_df,imgCnnTempo_df, how='left', on='Midi_name')
score_df = score_df.fillna({'Composer_cnn_score' : 0})

# 필요없는 열 제거
score_df = score_df.drop([ 'Genre_name', 'key_signature', 'harmonic_reduction' ], axis=1)
Labels = ['Bach', 'Mozart', 'Chopin', 'Clementi', 'Debussy', 'Brahms', 'Ravel', 'Mendelssohn', 'Beethoven', 'Schumann',
           'Classic', 'Gospel', 'Jazz', 'New_Age', 'POP', 'RagTime', 'Users',
           'Dreamy', 'Formulaic', 'Grand', 'Happy', 'Relaxed', 'Sad',
           'Ballade', 'Folk', 'Mazurka', 'Nocturne', 'Prelude', 'Scherzo', 'Waltz',
           'Autumn', 'Coffee', 'Morning', 'Night', 'Romance', 'Summer', 'Winter', 'Working',
           'Allegretto', 'Andante', 'lento', 'Vivace'
        ]

for i in Labels:
    i = i + '_s'    
    score_df = score_df.drop([i], axis=1)

score_df = score_df.drop(['Composer_cnn_best', 'Composer_cnn_best_score'], axis=1)
score_df = score_df.drop(['Genre_cnn_best', 'Genre_cnn_best_score'], axis=1)
score_df = score_df.drop(['mood_cnn_best', 'mood_cnn_best_score'], axis=1)
score_df = score_df.drop(['Rhythm_cnn_best', 'Rhythm_cnn_best_score'], axis=1)
score_df = score_df.drop(['tag_cnn_best', 'tag_cnn_best_score'], axis=1)
score_df = score_df.drop(['tempo_cnn_best', 'tempo_cnn_best_score'], axis=1)


# 점수합산
score_df['total_score'] = score_df['harmonic_reduction_score'] + \
                          score_df['key_sim_score'] + \
                          score_df['Composer_cnn_score'] + \
                          score_df['Genre_cnn_score'] + \
                          score_df['mood_cnn_score'] + \
                          score_df['Rhythm_cnn_score'] + \
                          score_df['tag_cnn_score'] + \
                          score_df['tempo_cnn_score']

# 점수 내림차순 정렬
score_df = score_df.sort_values('total_score', ascending=False)
score_df = score_df.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')
# save
score_df.to_pickle(path + 'total_score_df.pkl')
print('\n\n')
print('score_df print')
print('\n\n')
print(score_df)


name_sim_df = score_df[score_df['Midi_name'].str.contains(search_song,case=False)]
print(name_sim_df)

# setting values
set_df = pd.read_excel(path + 'setting_df.xlsx')
set_df = set_df.drop([ 'Unnamed: 0', 'Genre_name', 'Filepath', 'Composer' ], axis=1)



# best score setting
best_score_Midi_name = score_df['Midi_name'].values[1]

best_score_set_df = set_df[ set_df.Midi_name == best_score_Midi_name]
print(best_score_set_df)
