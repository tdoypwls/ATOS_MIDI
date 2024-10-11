import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from music21 import converter, corpus, instrument, midi, note, chord, pitch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from music21 import stream
from music21 import roman
from multiprocessing import Pool
from multiprocessing import Process, freeze_support
from pathlib import Path
import gensim, logging #gensim 은 python3 기준 4.0이상
import pprint
import pickle # pip3 install pickle5 (파이썬3.8 이상은 5로 설치해야함)
# pickle pip 오류 시 https://needneo.tistory.com/83 참고. C+ 14버전 업데이트 필요
# https://visualstudio.microsoft.com/visual-cpp-build-tools/ 


from setup_midi import setup, midiplt, harmonic



if __name__ == '__main__':

    # import harmonic_reduction setting (Task) 
    psd_df = pd.read_pickle('./dataframe/program_set_df.pkl')
    dataSet_df_Update_ALL = bool(psd_df['harmocin_reduction_dataSet_df_Update_ALL'].values[0])
    dataSet_df_Update_Users = bool(psd_df['harmocin_reduction_dataSet_df_Update_Users'].values[0])
    merge_df_read = bool(psd_df['harmocin_reduction_merge_df_read'].values[0])
    isWindows = bool(psd_df['isWindows'].values[0])

    #dataSet_df_Update_ALL = False
    #dataSet_df_Update_Users = False
    #merge_df_read = False
    #isWindows = True

    # 외부 클래스 지정
    stm = setup()
    mplt = midiplt()
    hm = harmonic()

    # Defining some constants and creating a new folder for MIDIs.
    midi_path = "./MIDIs"
    player_folder = "Player_MIDI/current_user"
    dataSet_folder = 'DataSet_MIDI'

    # Some helper methods.    
    player_path = stm.concat_path(midi_path, player_folder)
    dataSet_path = stm.concat_path(midi_path, dataSet_folder)

    # base_midi 불러오기
    base_midi_file = ''.join(os.listdir(player_path))
    print('current base_midi_file_name : ', base_midi_file)
    base_midi = stm.open_midi(stm.concat_path(player_path, base_midi_file), True)

    # 조성 분석
    timeSignature = base_midi.getTimeSignatures()[0]
    music_analysis = base_midi.analyze('key')
    print("Music time signature: {0}/{1}".format(timeSignature.beatCount, timeSignature.denominator))
    print("Expected music key: {0}".format(music_analysis))
    print("Music key confidence: {0}".format(music_analysis.correlationCoefficient))
    print("Other music key alternatives:")
    for analysis in music_analysis.alternateInterpretations:
        if (analysis.correlationCoefficient > 0.5):
            print(analysis)

    # MIDI 파일 악기 구성 출력
    print(stm.list_instruments(base_midi)) 

    # Focusing only on 6 first measures to make it easier to understand.
    #mplt.print_parts_countour(base_midi.measures(0, 6))

    # Pitch Class Histogram 많이 분포되어있는 화성
    #base_midi.plot('histogram', 'pitchClass', 'count')

    # Pitch Class by Offset Scatter 시간에 따라 중간에 조성이 바뀌는지 확인
    #base_midi.plot('scatter', 'offset', 'pitchClass')

    # Music reduction (곡의 변형, 복잡한 리듬, block 코드등을 배제하여 복잡도 감소)
    temp_midi_chords = stm.open_midi(
        stm.concat_path(player_path, base_midi_file),
        True).chordify()
    temp_midi = stream.Score()
    temp_midi.insert(0, temp_midi_chords)

    # Printing merged tracks.
    #mplt.print_parts_countour(temp_midi)
    # Dumping first measure notes
    #temp_midi_chords.measures(0, 1).show("text")

    # harmonic_reduction
    print(hm.harmonic_reduction(base_midi)[0:10])

    # 음악 장르, 폴더 생성
    #이미지데이터의 경로와 label데이터로 데이터프레임 만들기 
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

    folderpath = './MIDIs/DataSet_MIDI/'
    target_genres = dict()
    for i in get_label(folderpath):
        target_genres[i] = i

    #print(target_genres.items())


    # music_df 불러오기
    df_path = './dataframe/'
    music_df = pd.read_pickle(df_path + 'harmonic_reduction_music_df.pkl')


    #print('main.py 현재 실행 상태는 ' + str(__name__) + '입니다.' )
    if dataSet_df_Update_ALL:
        music_df = hm.create_midi_dataframe(dataSet_path, target_genres)
        music_df.to_pickle(df_path + 'harmonic_reduction_music_df.pkl')
        None
    else:
        None

    if dataSet_df_Update_Users:
        # users 폴더 추가 (DataSet 폴더 업데이트 한 경우 생략 가능
        # 유저 폴더에 새로 곡 추가할 경우만 실행
        user_path = './MIDIs/DataSet_MIDI/Users'
        user_music_df = hm.create_single_midi_dataframe(user_path)
        user_music_df.to_pickle(df_path + 'harmonic_reduction_user_music_df.pkl')
    else:
        None

    # import current_user_music_df
    current_user_path = './MIDIs/Player_MIDI/current_user'
    current_user_music_df = hm.create_single_midi_dataframe(current_user_path)
    current_user_music_df.to_pickle(df_path + 'harmonic_reduction_current_user_music_df.pkl')
    print('current_user_music_df')
    print(current_user_music_df)
    # key hist plt 6x6 출력
    #print(str(midiplt.key_hist_show(music_df, target_genres)))



    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(music_df["harmonic_reduction"], min_count=2, window=4)
  
    print("List of chords found:")
    #print(model.wv.key_to_index.keys())
    print("Number of chords considered by model: {0}".format(len(model.wv.key_to_index)))
 
    hm.get_related_chords(model, 'I')
    hm.get_related_chords(model, 'iv')
    hm.get_related_chords(model, 'V')

    # The first one should be smaller since "i" and "ii" chord doesn't share notes,
    # different from "IV" and "vi" which share 2 notes.
    hm.get_chord_similarity(model, "I", "ii") 
    hm.get_chord_similarity(model, "IV", "vi")

    # This one should be bigger because they are "enharmonic".
    # 아래 코드 갑자기오류나서 해결중, 일단 Pass (Key -i not present)
    #hm.get_chord_similarity(model, "-i", "vii")



    # merge df
    if merge_df_read:
        merge_df = pd.read_pickle(df_path + 'harmonic_reduction_merge_df.pkl')

    else:
        merge_df = pd.concat([music_df,current_user_music_df]) # 유저 샘플 데이터프레임과, 데이터셋 데이터프레임을 병합
        if dataSet_df_Update_Users:
            merge_df = pd.concat([merge_df,user_music_df]) #유저 데이터셋이 새로 업데이트 됐을 경우만, DataSet 전체 업데이트를 한 경우 생략
        merge_df.drop_duplicates(['Midi_name','Genre_name']) # 중복데이터 제거

        # save merge_df
        merge_df.to_pickle(df_path + 'harmonic_reduction_merge_df.pkl')


    print('print merge_df')
    print(merge_df)


    # 분석할 샘플 데이터 파일명 기입
    current_midi_file_name = current_user_music_df['Midi_name'].values[0]
    sim_genre = ""          # ""는 모든 장르에 대해서 비교
    sim_df = hm.calculate_similarity(merge_df, model, current_midi_file_name, sim_genre)
    sim_df = pd.DataFrame(data=sim_df)

    # save sim_df
    sim_df.to_pickle(df_path + 'harmonic_reduction_sim_df.pkl')
    print('sim_df')
    print(sim_df)


    # 특정 조건 검색
    #cont_name = 'SY'
    cont_name = str(psd_df['cont_name'].values[0])

    name_sim_df = sim_df[sim_df['Midi_name'].str.contains(cont_name,case=False)]
    print('name_sim_df')
    print(name_sim_df)

    # Best Score 
    best_score = hm.best_score_ck(merge_df, model, current_midi_file_name)
    print('best Score')
    print(best_score)
