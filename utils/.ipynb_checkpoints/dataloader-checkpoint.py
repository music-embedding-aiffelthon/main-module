from torch.utils.data import Dataset
import os
import numpy as np
import json
from tqdm import tqdm
import pickle

class mel_dataset(Dataset):

    def __init__(self, data_dir, genre_num):
        
        super(mel_dataset, self).__init__()
        
        
        meta_file_path = os.path.join(data_dir, "song_meta.json")
        if os.path.isfile(meta_file_path):
            with open(meta_file_path) as f:
                song_meta = json.load(f)
        else:
            raise FileNotFoundError(f'No such file or directory: {data_dir}/song_meta.json')
        
        song_dict = {}
        genre_dict = {}
        print('Load song_meta.json...')
        for song in tqdm(song_meta):
            song_dict[str(song['id'])] = song['song_gn_gnr_basket']
            for i in song['song_gn_gnr_basket']:
                try:
                    genre_dict[i] += 1
                except:
                    genre_dict[i] = 0
                    
        self.genre_count = {k:v for k,v in genre_dict.items()}
        self.genre_keys = {k:v for k,v in enumerate(list(genre_dict.keys()))}
        self.genre_index = {k:v for v,k in enumerate(list(genre_dict.keys()))}
        
        result_dict = {}        
        print('Load complete!')
        print('\nLoad file list...')
        for roots, dirs, files in tqdm(os.walk(data_dir)):
                  
            listdir = [os.path.join(roots, file) for file in files]
            for i in listdir:
                
                if ".pickle" in i:
                    with open(i, 'rb') as handle:
                        b = pickle.load(handle)
                    if b.shape[1] != 1876:
                        pass
                    else: 
                        try:
                            song_id = i.split('/')[-1].replace('.pickle','')
                            result_dict[i] = song_dict[song_id]
                        except:
                            print(song_id,'passed.')
                  
        file_list = []
        label = []
        
            
        for song_id, genres in result_dict.items():
            if len(genres) == 1:
                if genre_num == 'total':
                    one_hot_zero = np.zeros(len(self.genre_index))                        
                    for value in genres:                    
                        one_hot_zero[self.genre_index[value]] = 1
                    file_list.append(song_id)
                    label.append(one_hot_zero)

                else: 
                    if genre_num >= len(self.genre_index):
                        raise ValueError(f"There's no {genre_num} index genre. Reduce genre index number.")
                    if self.genre_keys[genre_num] in genres:
                        one_hot_zero = np.zeros(len(self.genre_index))                        
                        for value in genres:                    
                            one_hot_zero[self.genre_index[value]] = 1
                        file_list.append(song_id)
                        label.append(one_hot_zero)
            else:
                pass


        self.file_list = file_list
        self.label = label
        
    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as handle:
            x = pickle.load(handle)
        self.x = x
        return self.x, self.label[index]
    
    def __len__(self):
        return len(self.file_list)
    
    def genre_index(self):
        return self.genre_index