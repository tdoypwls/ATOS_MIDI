# 터미널 입력 (가상환경 설정) conda activate projectNew

# install list
# conda update -n base -c defaults conda
# conda update pip

import subprocess
import sys
from pandas import Series, DataFrame
from tensorflow.python.client import device_lib


# tensorflow GPU test
# https://doitgrow.com/m/28
print('Check tensorflow GPU')
#print(device_lib.list_local_devices())
print('\n\n')


# program setting

psd = {}
psd['cont_name'] = ['CH']

# harmonic_reduction_similarity.py control
psd['MIDI_device_autoSelect(0)'] = [False]
psd['harmocin_reduction_dataSet_df_Update_ALL'] = [False]
psd['harmocin_reduction_dataSet_df_Update_Users'] = [False]
psd['harmocin_reduction_merge_df_read'] = [False]
psd['isWindows'] = [True]

# midi to img control
psd['dataSet_to_img'] = [False]
psd['users_to_img'] = [True]

psd['composer_to_img'] = [False]
psd['genre_to_img'] = [False]
psd['mood_to_img'] = [False]
psd['rhythm_to_img'] = [False]
psd['tag_to_img'] = [False]
psd['tempo_to_img'] = [False]

# img_cnn_for_composer.py control
psd['icfc_isNewModelTrain'] = [False]
psd['icfc_isDFPC_DataSet'] = [False]
psd['icfc_isDFPC_users'] =  [True]
psd['icfc_isDFPC_composer'] = [False]
psd['icfc_isDFPC_genre'] = [False]
psd['icfc_isDFPC_mood'] = [False]
psd['icfc_isDFPC_rhythm'] = [False]
psd['icfc_isDFPC_tag'] = [False]
psd['icfc_isDFPC_tempo'] = [False]

# img_cnn_for_genre.py control
psd['icfg_isNewModelTrain'] = [False]
psd['icfg_isDFPC_DataSet'] = [False]
psd['icfg_isDFPC_users'] =  [True]
psd['icfg_isDFPC_composer'] = [False]
psd['icfg_isDFPC_genre'] = [False]
psd['icfg_isDFPC_mood'] = [False]
psd['icfg_isDFPC_rhythm'] = [False]
psd['icfg_isDFPC_tag'] = [False]
psd['icfg_isDFPC_tempo'] = [False]

# img_cnn_for_mood.py control
psd['icfm_isNewModelTrain'] = [False]
psd['icfm_isDFPC_DataSet'] = [False]
psd['icfm_isDFPC_users'] =  [True]
psd['icfm_isDFPC_composer'] = [False]
psd['icfm_isDFPC_genre'] = [False]
psd['icfm_isDFPC_mood'] = [False]
psd['icfm_isDFPC_rhythm'] = [False]
psd['icfm_isDFPC_tag'] = [False]
psd['icfm_isDFPC_tempo'] = [False]

# img_cnn_for_rhythm.py control
psd['icfr_isNewModelTrain'] = [False]
psd['icfr_isDFPC_DataSet'] = [False]
psd['icfr_isDFPC_users'] =  [True]
psd['icfr_isDFPC_composer'] = [False]
psd['icfr_isDFPC_genre'] = [False]
psd['icfr_isDFPC_mood'] = [False]
psd['icfr_isDFPC_rhythm'] = [False]
psd['icfr_isDFPC_tag'] = [False]
psd['icfr_isDFPC_tempo'] = [False]

# img_cnn_for_tag.py control
psd['icft_isNewModelTrain'] = [False]
psd['icft_isDFPC_DataSet'] = [False]
psd['icft_isDFPC_users'] =  [True]
psd['icft_isDFPC_composer'] = [False]
psd['icft_isDFPC_genre'] = [False]
psd['icft_isDFPC_mood'] = [False]
psd['icft_isDFPC_rhythm'] = [False]
psd['icft_isDFPC_tag'] = [False]
psd['icft_isDFPC_tempo'] = [False]

# img_cnn_for_tempo.py control
psd['icftp_isNewModelTrain'] = [False]
psd['icftp_isDFPC_DataSet'] = [False]
psd['icftp_isDFPC_users'] =  [True]
psd['icftp_isDFPC_composer'] = [False]
psd['icftp_isDFPC_genre'] = [False]
psd['icftp_isDFPC_mood'] = [False]
psd['icftp_isDFPC_rhythm'] = [False]
psd['icftp_isDFPC_tag'] = [False]
psd['icftp_isDFPC_tempo'] = [False]


# merge_to_dataframe.py contrl
psd['key_signature_ALL'] = [True] # True일경우 조성 전체가 일치해야 가산점 부여, False는 major minora

psd['composer_correction'] = [0.12]
psd['composer_correction_na'] = [- 0.04]
psd['composer_correction_count'] = [0]

psd['genre_correction'] = [0.12]
psd['genre_correction_na'] = [- 0.04]
psd['genre_correction_count'] = [0]

psd['mood_correction'] = [0.12]
psd['mood_correction_na'] = [- 0.04]
psd['mood_correction_count'] = [0]

psd['rhythm_correction'] = [0.12]
psd['rhythm_correction_na'] = [- 0.04]
psd['rhythm_correction_count'] = [0]

psd['tag_correction'] = [0.12]
psd['tag_correction_na'] = [- 0.04]
psd['tag_correction_count'] = [0]

psd['tempo_correction'] = [0.12]
psd['tempo_correction_na'] = [- 0.04]
psd['tempo_correction_count'] = [0]

psd_df = DataFrame(psd)
psd_df.to_pickle('./dataframe/program_set_df.pkl')



# loop file test 작성중
test_folder_path = "./MIDIs/Test_UserDataSet/"




# start programs

try:
    print('1. recordinng midi')
    print('\n\n')
    #n1 = subprocess.run(args=[sys.executable, './MIDI_recorder/recorder.py'])

except KeyboardInterrupt:
    print('\n\n')

finally:
    print('2. harmonic reduction')
    print('\n\n')
    n2 = subprocess.run(args=[sys.executable, 'harmonic_reduction_similarity.py'])
    print('\n\n')

    print('3. midi_to_img')
    print('\n\n')
    n3 = subprocess.run(args=[sys.executable, 'midi_to_img.py'])
    print('\n\n')

    print('4. img cnn for composer')
    print('\n\n')
    n4 = subprocess.run(args=[sys.executable, 'img_cnn_for_composer.py'])
    print('\n\n')

    print('5. img cnn for genre')
    print('\n\n')
    n4 = subprocess.run(args=[sys.executable, 'img_cnn_for_genre.py'])
    print('\n\n')

    print('6. img cnn for mood')
    print('\n\n')
    n4 = subprocess.run(args=[sys.executable, 'img_cnn_for_mood.py'])
    print('\n\n')

    print('7. img cnn for rhythm')
    print('\n\n')
    n5 = subprocess.run(args=[sys.executable, 'img_cnn_for_rhythm.py'])
    print('\n\n')

    print('8. img cnn for tag')
    print('\n\n')
    n5 = subprocess.run(args=[sys.executable, 'img_cnn_for_tag.py'])
    print('\n\n')

    print('9. img cnn for tempo')
    print('\n\n')
    n5 = subprocess.run(args=[sys.executable, 'img_cnn_for_tempo.py'])
    print('\n\n')

    print('10. merge to dataframe')
    print('\n\n')
    n6 = subprocess.run(args=[sys.executable, 'merge_to_dataframe.py'])
    print('\n\n')

    print('11. save user settinng')
    print('\n\n')
    # n7 = subprocess.run(args=[sys.executable, 'save_user_setting.py'])

    print('=== End ===')

