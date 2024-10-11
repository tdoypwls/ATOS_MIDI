import pandas as pd
"""
path = './dataframe/harmonic_reduction_sim_df.pkl'
df = pd.read_pickle(path)
df = df.sort_values('harmonic_reduction_score',ascending=False)
print(df)


cont_name = input('serch file name :')
name_sim_df = df[df['Midi_name'].str.contains(cont_name,case=False)]
print('name_sim_df')
print(name_sim_df)
print('\n\n\n\n')
"""

path = './dataframe/total_score_df.pkl'
df = pd.read_pickle(path)
df = df.sort_values('total_score',ascending=False)
df = df.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')
print(df)


#cont_name = input('serch file name :')
#name_sim_df = df[df['Midi_name'].str.contains(cont_name,case=False)]
name_sim_df = df[df['Composer_cnn_score'].str.contains(0.0,case=False)]
print('name_sim_df')
print(name_sim_df)
print('\n\n\n\n')