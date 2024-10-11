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
import gensim, logging #gensim 은 python3 기준 4.0이상
import pprint
import pickle # pip3 install pickle5 (파이썬3.8 이상은 5로 설치해야함)
# pickle pip 오류 시 https://needneo.tistory.com/83 참고. C+ 14버전 업데이트 필요
# https://visualstudio.microsoft.com/visual-cpp-build-tools/ 




class setup:
    
    def concat_path(self, path, child): # 폴더 경로 합치기
        return path + "/" + child

    def open_midi(self, midi_path, remove_drums): # 미디 열기
        # There is an one-line method to read MIDIs
        # but to remove the drums we need to manipulate some
        # low level MIDI events.
        mf = midi.MidiFile()
        mf.open(midi_path)
        mf.read()
        mf.close()
        if (remove_drums):
            for i in range(len(mf.tracks)):
                mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]          

        return midi.translate.midiFileToStream(mf)
    
    def get_file_name(self, link):
        filename = link.split('/')[::-1][0]
        return filename

    def list_instruments(self, midi): # MIDI파일 내 악기 리스트 출력
        partStream = midi.parts.stream()
        print("List of instruments found on MIDI file:")
        for p in partStream:
            aux = p
            print (p.partName)

class midiplt:

    # 미디파일 plot 표시
    def extract_notes(self, midi_part):
        parent_element = []
        ret = []
        for nt in midi_part.flat.notes:        
            if isinstance(nt, note.Note):
                ret.append(max(0.0, nt.pitch.ps))
                parent_element.append(nt)
            elif isinstance(nt, chord.Chord):
                for pitch in nt.pitches:
                    ret.append(max(0.0, pitch.ps))
                    parent_element.append(nt)
        
        return ret, parent_element

    def print_parts_countour(self, midi):

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)
        minPitch = pitch.Pitch('C10').ps
        maxPitch = 0
        xMax = 0
        
        # Drawing notes.
        for i in range(len(midi.parts)):
            top = midi.parts[i].flat.notes                  
            y, parent_element = self.extract_notes(top)
            if (len(y) < 1): continue
                
            x = [n.offset for n in parent_element]
            ax.scatter(x, y, alpha=0.6, s=7)
            
            aux = min(y)
            if (aux < minPitch): minPitch = aux
                
            aux = max(y)
            if (aux > maxPitch): maxPitch = aux
                
            aux = max(x)
            if (aux > xMax): xMax = aux
        
        for i in range(1, 10):
            linePitch = pitch.Pitch('C{0}'.format(i)).ps
            if (linePitch > minPitch and linePitch < maxPitch):
                ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))            

        plt.ylabel("Note index (each octave has 12 notes)")
        plt.xlabel("Number of quarter notes (beats)")
        plt.title('Voices motion approximation, each color is a different instrument, red lines show each octave')
        plt.show()



    def key_hist(self, df, genre_name, ax):
        title = "All Musics Key Signatures"
        filtered_df = df
        if (genre_name is not None):
            title = genre_name + " Key Signatures"
            filtered_df = df[df["Genre_name"] == genre_name]
            
        return  filtered_df["key_signature"].value_counts().plot(ax = ax, kind='bar', title = title)


    def key_hist_show(self, df, target_genres):
        i = 1
        key_hist_list =[]
        fig, axes = plt.subplots(nrows=int(len(target_genres)/3) + 1, ncols = 3, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        for key, value in target_genres.items():
            key_hist_list.append(self.key_hist(df, key, axes[int(i/3), i%3]))
            i = i + 1
        
        return key_hist_list


class harmonic:
    
    def __init__(self):
        self.stm = setup()

    def note_count(self, measure, count_dict):
        bass_note = None
        for chord in measure.recurse().getElementsByClass('Chord'):
            # All notes have the same length of its chord parent.
            note_length = chord.quarterLength
            for note in chord.pitches:          
                # If note is "C5", note.name is "C". We use "C5"
                # style to be able to detect more precise inversions.
                note_name = str(note) 
                if (bass_note is None or bass_note.ps > note.ps):
                    bass_note = note
                    
                if note_name in count_dict:
                    count_dict[note_name] += note_length
                else:
                    count_dict[note_name] = note_length
            
        return bass_note
                    
    def simplify_roman_name(self, roman_numeral):
        # Chords can get nasty names as "bII#86#6#5",
        # in this method we try to simplify names, even if it ends in
        # a different chord to reduce the chord vocabulary and display
        # chord function clearer.
        ret = roman_numeral.romanNumeral
        inversion_name = None
        inversion = roman_numeral.inversion()
        
        # Checking valid inversions.
        if ((roman_numeral.isTriad() and inversion < 3) or
                (inversion < 4 and
                    (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
            inversion_name = roman_numeral.inversionName()
            
        if (inversion_name is not None):
            ret = ret + str(inversion_name)
            
        elif (roman_numeral.isDominantSeventh()): ret = ret + "M7"
        elif (roman_numeral.isDiminishedSeventh()): ret = ret + "o7"
        return ret
                    
    def harmonic_reduction(self, midi_file):
        ret = []
        temp_midi = stream.Score()
        temp_midi_chords = midi_file.chordify()
        temp_midi.insert(0, temp_midi_chords)    
        music_key = temp_midi.analyze('key')
        max_notes_per_chord = 4   
        for m in temp_midi_chords.measures(0, None): # None = get all measures.
            if (type(m) != stream.Measure):
                continue
            
            # Here we count all notes length in each measure,
            # get the most frequent ones and try to create a chord with them.
            count_dict = dict()
            bass_note = self.note_count(m, count_dict)
            if (len(count_dict) < 1):
                ret.append("-") # Empty measure
                continue
            
            sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
            sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
            measure_chord = chord.Chord(sorted_notes)
            
            # Convert the chord to the functional roman representation
            # to make its information independent of the music key.
            roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
            ret.append(self.simplify_roman_name(roman_numeral))
            
        return ret

    def process_single_file(self, midi_param):
        try:
            genre_name = midi_param[0]
            midi_path = midi_param[1]
            midi_name = self.stm.get_file_name(midi_path)
            midi = self.stm.open_midi(midi_path, True)

            print('input name(' + midi_name + ')')
            return (
                midi.analyze('key'),
                genre_name,
                self.harmonic_reduction(midi),
                midi_name)
        except Exception as e: # 에러처리
            print("Error on {0}".format(midi_name))
            print(e)
            return None

    
    def create_midi_dataframe(self, dataSet_path, target_genres):
        key_signature_column = []
        genre_name_column = []
        harmonic_reduction_column = []
        midi_name_column = []
        pool = Pool(8)
        midi_params = []
        for key, value in target_genres.items():
            folder_path = self.stm.concat_path(dataSet_path, key)
            print('input folder :' + str(key))
            for midi_name in os.listdir(folder_path):
                midi_params.append((key, self.stm.concat_path(folder_path, midi_name)))
                #print('midi_name_ck')
        print('number of files :' + str(len(midi_params)))

        results = pool.map(self.process_single_file, midi_params)
        for result in results:
            if (result is None):
                continue
                
            key_signature_column.append(result[0])
            genre_name_column.append(result[1])
            harmonic_reduction_column.append(result[2])
            midi_name_column.append(result[3])
        
        d = {'Midi_name': midi_name_column,
            'Genre_name': genre_name_column,
            'key_signature' : key_signature_column,
            'harmonic_reduction': harmonic_reduction_column}
        return pd.DataFrame(data=d)


    def create_single_midi_dataframe(self, folder_path):
        key_signature_column = []
        genre_name_column = []
        harmonic_reduction_column = []
        midi_name_column = []
        pool = Pool(6)
        midi_params = []
        
        print('input folder :' + str('Users'))
        for midi_name in os.listdir(folder_path):
            midi_params.append(('Users', self.stm.concat_path(folder_path, midi_name)))
            #print('midi_name_ck')

        print('number of files :' + str(len(midi_params)))
        
        results = pool.map(self.process_single_file, midi_params)
        for result in results:
            if (result is None):
                continue
                
            key_signature_column.append(result[0])
            genre_name_column.append(result[1])
            harmonic_reduction_column.append(result[2])
            midi_name_column.append(result[3])
        
        d = {'Midi_name': midi_name_column,
            'Genre_name': genre_name_column,
            'key_signature' : key_signature_column,
            'harmonic_reduction': harmonic_reduction_column}
        return pd.DataFrame(data=d)




    def get_related_chords(self, model, token, topn=3):
        print("Similar chords with " + token)
        for word, similarity in model.wv.most_similar(positive=[token], topn=topn):
            print (word, round(similarity, 3))

    def get_chord_similarity(self, model, chordA, chordB):
        print("Similarity between {0} and {1}: {2}".format(
            chordA, chordB, model.wv.similarity(chordA, chordB)))




    def vectorize_harmony(self, model, harmonic_reduction):
        # Gets the model vector values for each chord from the reduction.
        word_vecs = []
        for word in harmonic_reduction:
            try:
                vec = model.wv[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass
        
        # Assuming that document vector is the mean of all the word vectors.
        return np.mean(word_vecs, axis=0)

    def cosine_similarity(self, vecA, vecB):
        # Find the similarity between two vectors based on the dot product.
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        
        return csim

    def calculate_similarity_aux(self, df, model, source_name, target_names=[], threshold=0):
        source_harmo = df[df["Midi_name"] == source_name]["harmonic_reduction"].values[0]
        source_vec = self.vectorize_harmony(model, source_harmo)    
        results = []
        for name in target_names:
            target_harmo = df[df["Midi_name"] == name]["harmonic_reduction"].values[0]
            if (len(target_harmo) == 0):
                continue
                
            target_vec = self.vectorize_harmony(model, target_harmo)       
            sim_score = self.cosine_similarity(source_vec, target_vec)
            if sim_score > threshold:
                results.append({
                    'Midi_name' : name,
                    'harmonic_reduction_score' : sim_score

                })
                    
        # Sort results by score in desc order
        results.sort(key=lambda k : k['harmonic_reduction_score'] , reverse=True)
        return results

    def calculate_similarity(self, df, model, source_name, target_prefix, threshold=0):
        source_midi_names = df[df["Midi_name"] == source_name]["Midi_name"].values
        if (len(source_midi_names) == 0):
            print("Invalid source name")
            return
        
        source_midi_name = source_midi_names[0]
        
        target_midi_names = df[df["Genre_name"].str.startswith(target_prefix)]["Midi_name"].values  
        if (len(target_midi_names) == 0):
            print("Invalid target prefix")
            return
        
        return self.calculate_similarity_aux(df, model, source_midi_name, target_midi_names, threshold)


    # Best Score 
    def best_score_ck(self, merge_df, model, current_midi_file_name):
        rank_score = self.calculate_similarity(merge_df, model, current_midi_file_name, "")
        best_score_name = rank_score[1]['Midi_name']
        best_score = rank_score[1]['harmonic_reduction_score']
        #print('best score name : ',best_score_name,)

        return best_score_name, best_score