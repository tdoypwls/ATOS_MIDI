import os

file_path = './MIDIs/Player_MIDI/userDataSet'
file_names = os.listdir(file_path)
file_names

i = 0
conv = 0

for name in file_names:
    src = os.path.join(file_path, name)
    
    if '.midi' in str(src):
        dst = str(name).replace(".midi",".MID")
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        conv += 1
    elif '.mid' in str(src):
        dst = str(name).replace(".mid",".MID")
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        conv += 1
    elif '.Mid' in str(src):
        dst = str(name).replace(".Mid",".MID")
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        conv += 1
    else:
        None
    i += 1


print( 'file num : ',i)
print( 'complet conv num : ',i)
