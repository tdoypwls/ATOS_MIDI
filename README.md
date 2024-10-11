# ATOS MIDI
ATOS MIDI : Automatic tone optimizing system for digital piano using MIDI similarity analysis

## 1. Abstract
- This paper presents an automatic calibration system that determines the optimal sound settings for a musical piece by analyzing MIDI signals generated from digital piano performances using word2vec and graph CNN.
- Existing MIDI analysis often relies on harmonic analysis through the Music21 library or focuses on building harmonic progression classification models based on MIDI time-series graphs, resulting in limited applicability in business contexts and insufficient differentiation of the nuanced characteristics perceived by listeners.

- Therefore, we propose a new CNN classification model by combining the idea of ​​measuring harmonic progression similarity via word2vec and the concept of pixelating MIDI time series graphs. The model integrates multidimensional component weights into the analysis results to gain a deeper understanding of the user's playing style and generate customized sound settings that approach human-like adaptability.

## 2. Installation
- microsoft c++ build tools
- python-rtmidi
- pandas
- mido
- opencv-python
- numpy
- matplotlib
- seaborn
- tensorflow
- keras
- gensim # >= 4.0
- music21
- pickle5
- scikit-learn
- scipy
- tqdm
- ipython

## 3. Datasets
- MIDI data used in this training is not disclosed due to licensing issues. We have included a MIDI_recorder code that receives the user's performance directly as input in real time and uses it as data, and you can create your own dataset using this.

## 4. Structure
![Structure](./img/1.png)

## 5. Model
### Step 1 : Key signature
![2](./img/2.png)
### Step 2 : Harmonic Reduction(word2vec)
![3](./img/3.png)
### Step 3 : MIDI to pixel image
![3](./img/4.png)
### Step 4 : Classification CNN (ResNet152V2)
- [Composer,Rhythm,Tempo, Mood, Genre,Tag ... etc] 
### Step 5 : Score calculation



## 6. Result
- Chopin Fantasie Impromptu Op.66 
    - origin : https://github.com/user-attachments/assets/07bbab80-d7aa-4f09-9cd1-1913874a6014
    - result : https://github.com/user-attachments/assets/f0ddd76b-1bb9-41f7-9f14-24005f8779d0
  
- A.Ginastera Danzas Argentinas
    - origin : https://github.com/user-attachments/assets/2669a8f1-4654-436d-a222-0efea6003262
    - result : https://github.com/user-attachments/assets/7d22ed1b-0fbd-4610-a07f-91c8a8c1d730
  
- Debussy - Arabesque No.1 in E major
    - origin : https://github.com/user-attachments/assets/3413e371-1deb-4f84-a015-602221fa81e3
    - result :




  
- Mozart: Piano Sonata
    - origin : 
    - result : 


## 7. References

- https://towardsdatascience.com/midi-music-data-extraction-using-music21-and-word2vec-on-kaggle-cb383261cd4e
- https://github.com/narcode/MIDI_recorder
- https://github.com/asigalov61/MIDI-Tempo-Detective
- https://github.com/ori-2019-siit/AIMusicGenerator
- https://content.iospress.com/articles/semantic-web/sw210446
- https://ieeexplore.ieee.org/document/8457704
