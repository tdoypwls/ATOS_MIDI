# png to midi

from music21 import note, chord, stream, midi
import numpy as np
from PIL import Image


def open_midi(midi_path, remove_drums): # 미디 열기
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

def load_image(path):
    img = Image.open(path)
    img.load()
    img = img.rotate(90)
    return np.asarray(img, dtype=np.uint8)


def make_pitches(path):
    data = load_image(path)
    offset = 10
    music = stream.Stream()

    height, width, channels = data.shape

    for channel in range(channels):
        notes = {}  # timestamp 0: [n1, n2]  n = (pitch, duration)
        active_pitches = {p: None for p in range(height)}  # for every pitch, timestamp key pitch : timestamp
        for w in range(width):
            column = data[:, w]

            for pitch in range(height):
                if column[pitch][channel] == 255:
                    if w not in notes.keys():
                        notes[w] = {}
                    if active_pitches[pitch] is None:
                        notes[w][pitch] = 1
                        active_pitches[pitch] = w
                    else:
                        notes[active_pitches[pitch]][pitch] += 1
                else:
                    active_pitches[pitch] = None

        for timestamp in sorted(notes.keys()):
            if len(notes[timestamp]) == 1:
                n = note.Note()
                n.pitch.ps = list(notes[timestamp].keys())[0] + 46
                n.duration.quarterLength = list(notes[timestamp].values())[0] / offset
                music.append(n)
            else:
                durations = {}

                for pitch, duration in notes[timestamp].items():
                    if duration not in durations.keys():
                        durations[duration] = [pitch+46]
                    else:
                        durations[duration].append(pitch+46)

                for duration, pitches in durations.items():
                    if len(pitches) == 1:
                        n = note.Note()
                        n.pitch.ps = pitches[0] + 46
                        n.duration.quarterLength = duration/offset
                        music.append(n)
                    else:
                        c = chord.Chord(pitches)
                        c.duration.quarterLength = duration/offset
                        music.append(c)

    return music


def save_midi(stream, path):
    name = path[path.rfind("/") + 1:path.rfind(".")]
    nameNum = name.split('.MID')[-1]
    name = name.replace(nameNum,'')
    print(name)
    stream.write("midi", "./png_to_midi/{}".format(name))


def image_to_midi(path):
    save_midi(make_pitches(path), path)

# png to midi

image_to_midi("./DFPC_users/current_user/current_user_song.MID_0.png")


base_midi = open_midi("./png_to_midi/current_user_song.MID", True)

# 악보출력 시 PC에서는 musecore 프로그램 설치 필요 https://musescore.org
#base_midi.measures(0, 8).show() # print score
#base_midi.measures(0, 4).show('midi')       # colab에서는 작동하지 않음 
#play(base_midi2.measures(0, 17)) # play midi   # colab에서 위 코드 대신 사용
