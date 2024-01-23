# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 08:18:55 2023

@author: VuralBayraklii
"""
import os
import pickle

class SavePickle:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def save_process(self):
        # "pickles" dizini yoksa oluştur
        if not os.path.exists('pickles'):
            os.makedirs('pickles')

        try:
            # Pickle dosyasını yaz
            with open(os.path.join('pickles', f'{self.name}.pkl'), 'wb') as f:
                pickle.dump(self.data, f)
            print(f"{self.name} successfully created.")
        except Exception as e:
            print(f"Error: {e}")