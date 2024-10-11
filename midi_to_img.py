
# MIDI to png task

from music21 import midi, note, chord
#from instruments_loader import load_from_bin
import fractions
import pickle
import numpy as np
from PIL import Image
import os
from joblib import Parallel, delayed
import time
import random
import glob
from pathlib import Path
import shutil
import pandas as pd 

# option
# 기본 및 main.py 실행 시  users_to_img만 True, 나머지 False로 설정해야함.
psd_df = pd.read_pickle('./dataframe/program_set_df.pkl')
dataSet_to_img = bool(psd_df['dataSet_to_img'].values[0])
users_to_img = bool(psd_df['users_to_img'].values[0])

composer_to_img = bool(psd_df['composer_to_img'].values[0])
genre_to_img = bool(psd_df['genre_to_img'].values[0])
mood_to_img = bool(psd_df['mood_to_img'].values[0])
rhythm_to_img = bool(psd_df['rhythm_to_img'].values[0])
tag_to_img = bool(psd_df['tag_to_img'].values[0])
tempo_to_img = bool(psd_df['tempo_to_img'].values[0])

#dataSet_to_img = False
#rhythm_to_img = False
#users_to_img = True
#composer_to_img = False
#genre_to_img = False


project_path = "./"



def load_from_bin(path):
    return pickle.load(open(path, "rb"))

def txt_to_bin():
    txt_file = open(os.path.join("instruments", "instruments_lstm.txt"), "r")
    sorted_instruments = parse_text_file(txt_file)
    pickle.dump(sorted_instruments, open(os.path.join("instruments", "instruments_lstm.bin"), "wb"))

def parse_text_file(f):
    sorted_instruments = []
    for line in f:
        sorted_instruments.append(line.split()[0])

    return sorted_instruments

def open_midi(midi_path, remove_drums=True):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)

def extract_line(midi_file, instruments_list):
    part_stream = midi_file.parts.stream()

    for instrument in instruments_list:
        for part in part_stream:
            if instrument == part.partName:
                pitches = extract_notes(part)
                return pitches

    # some midi files have no instrument names, so sample the first part
    pitches = extract_notes(part_stream[0])
    return pitches

def extract_notes(part):
    pitches = []

    for nt in part.flat.notes:
        if isinstance(nt, note.Note):
            duration = nt.duration.quarterLength
            if isinstance(duration, fractions.Fraction):
                duration = round(float(duration), 1)
            pitches.append((nt.pitch.ps, duration))
        elif isinstance(nt, chord.Chord):
            akord = []
            duration = nt.duration.quarterLength
            if isinstance(duration, fractions.Fraction):
                duration = round(float(duration), 1)
            for pitch in nt.pitches:
                akord.append((pitch.ps, duration))
            pitches.append(akord)

    return pitches

def make_image(pitches, name):
    counter = 0
    width, height = 64, 64
    data = np.zeros((height, width, 3), dtype=np.uint8)
    start = 0
    offset = 10
    channel = 0
    images = []

    for element in pitches:
        notes = []
        if isinstance(element, list):
            # take first note from chord and it's duration, calculate number of sequential pixels needed
            pixels = int(element[0][1] * offset)
            for note in element:
                notes.append(int(note[0]))
        else:
            # equivalent for single note
            pixels = int(element[1] * offset)
            notes.append(int(element[0]))

        # check if it can fit on image
        if not start + pixels < width:
            channel += 1
            start = 0
            # we have filled every channel and need to return a value
            if channel == 3:
                img = Image.fromarray(data, "RGB")
                img = img.rotate(-90)
                images.append(img)
                data = np.zeros((height, width, 3), dtype=np.uint8)
                start = 0
                channel = 0
                counter += 1

        for note in notes:
            ps = data[note-46][start:start + pixels + 1] # max pitch is 93, min is 48
            for p in ps:
                p[channel] = 255
        start = start + pixels + 1

    return images

def do_work(path, instruments):
    try:
        name = path[path.rfind("\\") + 1:path.rfind(".")]
        print(name)
        song = open_midi(path)
        pitches = extract_line(song, instruments)
        return make_image(pitches, name)
    except Exception as e:
        print(str(e))
        return None

def get_input_paths(path):
    paths = []

    for root, dirs, files in os.walk(path):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths

def make_all_images(paths, num_cores, instruments):
    results = Parallel(n_jobs=num_cores)(delayed(do_work)(p, instruments) for p in paths)
    return results

def find_ps_range(paths, instruments):
    maksimum = 0
    minimum = 1000
    from operator import itemgetter
    for path in paths:
        try:
            song = open_midi(path)
            pitches = extract_line(song, instruments)
            flat_list = [item for sublist in pitches for item in sublist]
            iter_minmum = min(flat_list, key=itemgetter(0))[0]
            iter_max = max(flat_list, key=itemgetter(0))[0]
            if iter_minmum < minimum:
                minimum = iter_minmum
            if iter_max > maksimum:
                maksimum = iter_max
        except:
            continue
    return minimum, maksimum

def midi_to_png_start(midi_path, folderName, fileName, dataSet_Classification):
    instruments = load_from_bin(os.path.join("./instruments", "instruments_lstm.bin"))
    #paths = get_input_paths(path)
    #random.shuffle(paths)
    paths = [midi_path + '/' + folderName + '/' + fileName ]
    #paths = [midi_path]
    print('Dataset path : ' + str(paths))
    print('Dataset name : ' + folderName)
    #print('Dataset len : ' + str(len(paths)))
    batch_size = 32
    counter = 0

    batches = [paths[i * batch_size:(i + 1) * batch_size] for i in range((len(paths) + batch_size - 1) // batch_size)]

    
    for batch in batches:
        results = make_all_images(batch, -1, instruments)
        #print('lmages len : ' + str(len(results)))
       #print(results)

        for song_images in results:

            if not song_images is None:
                for part in song_images:
                    name = fileName + '_' + str(counter) + ".png"
                    #print(name)
                    path_test = './' + dataSet_Classification + '/' + folderName
                    path_train = './' + dataSet_Classification + '/' + folderName
                    if counter % 10 == 0:
                        if os.path.isdir(path_test):
                          None
                        else:
                          os.makedirs(path_test)
                        part.save(path_test + '/' + name)
                    else:
                        if os.path.isdir(path_train):
                          None  
                        else:
                          os.makedirs(path_train)
                        part.save(path_train + '/' + name)
                    counter += 1
                print('midi to png len : ' + str(counter))

## midi to png 생성
def midi_to_png_main(dir_path,dataSet_Classification):

    pathfolder = Path(dir_path)
    filepaths = list(pathfolder.glob(r'**/*.MID'))

    # window만 아래 과정 필요====================
    filepathsStr = []
    for i in filepaths:
        filepathsStr.append(str(i).replace('\\','/'))
    filepaths = filepathsStr
    # windows 과정 끝 ===================
    


    # 윈도우 아니면 아래 코드 활성화
    #filepaths2 = list(pathfolder.glob(r'**/*.mid'))
    #filepaths = filepaths + filepaths2
    #print(len(filepaths))



    #print(filepaths)
    labels_all = [str(filepaths[i]).split("/")[-2] \
              for i in range(len(filepaths))]

    folderNames = [] # 중복 제거된 값들이 들어갈 리스트

    for value in labels_all:
        if value not in folderNames:
            folderNames.append(value)
            #print(value)
    #print(folderNames)

    start_time = time.time()
    pathfolder_str = str(pathfolder)

    for folderName in folderNames:
        pathfolder_MID = Path(dir_path + folderName) 
        filepaths_MID = list(pathfolder_MID.glob(r'**/*.MID'))

        # window만 아래 과정 필요===================
        filepaths_MIDStr = []
        for i in filepaths_MID:
            filepaths_MIDStr.append(str(i).replace('\\','/'))
        filepaths_MID = filepaths_MIDStr
        #print(filepaths_MID)
        # windows 과정 끝 ===================


        # 윈도우 아니면 아래 코드 활성화
        #filepaths_MID2 = list(pathfolder_MID.glob(r'**/*.mid'))
        #filepaths_MID = filepaths_MID + filepaths_MID2


        file_labels_all = [str(filepaths_MID[i]).split("/")[-1] \
              for i in range(len(filepaths_MID))]

        fileNames = []
        for value in file_labels_all:
            if value not in fileNames:
                fileNames.append(value)
                #print(value)
      #print(fileNames)

        #print(folderName)
        for fileName in fileNames:
            midi_to_png_start(pathfolder_str,folderName, fileName, dataSet_Classification)
    print("Time elapsed: {:0.2f} seconds".format(time.time() - start_time))


if dataSet_to_img:
    shutil_dir_path = project_path + 'DFPC_DataSet' # DFPC 폴더 초기화 (내용물 삭제)
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)
    dataSet_dir_path = './MIDIs/DataSet_MIDI/'
    midi_to_png_main(dataSet_dir_path, 'DFPC_DataSet')

if users_to_img:
    shutil_dir_path = project_path + 'DFPC_users'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)
    user_dir_path = './MIDIs/Player_MIDI/'
    midi_to_png_main(user_dir_path, 'DFPC_users')

if composer_to_img:
    shutil_dir_path = project_path + 'DFPC_composer'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)
    composer_dir_path = './MIDIs/MDS_Composer/'
    midi_to_png_main(composer_dir_path,'DFPC_composer')

if genre_to_img:
    shutil_dir_path = project_path + 'DFPC_genre'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)    
    genre_dir_path = './MIDIs/MDS_Genre/'
    midi_to_png_main(genre_dir_path, 'DFPC_genre')

if mood_to_img:
    shutil_dir_path = project_path + 'DFPC_mood'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)    
    mood_dir_path = './MIDIs/MDS_Mood/'
    midi_to_png_main(mood_dir_path, 'DFPC_mood')

if rhythm_to_img:
    shutil_dir_path = project_path + 'DFPC_rhythm'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)
    rhythm_dir_path = './MIDIs/MDS_Rhythm/'
    midi_to_png_main(rhythm_dir_path, 'DFPC_rhythm')

if tag_to_img:
    shutil_dir_path = project_path + 'DFPC_tag'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)
    tag_dir_path = './MIDIs/MDS_Tag/'
    midi_to_png_main(tag_dir_path, 'DFPC_tag')

if tempo_to_img:
    shutil_dir_path = project_path + 'DFPC_tempo'
    if os.path.exists(shutil_dir_path):
        shutil.rmtree(shutil_dir_path)
    tempo_dir_path = './MIDIs/MDS_Tempo/'
    midi_to_png_main(tempo_dir_path, 'DFPC_tempo')