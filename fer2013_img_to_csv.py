#encoding:utf-8
import pandas as pd
import numpy as np
import cv2
import os
import progressbar
 
emotions = {
    'anger':'0', #生气
    'disgust':'1', #厌恶
    'fear':'2', #恐惧
    'happy':'3', #开心
    'sad':'4', #伤心
    'surprised':'5', #惊讶
    'normal':'6', #中性
}

selected_emotions = {
    'anger':'0', #生气
    'disgust':'1', #厌恶
    'happy':'2', #开心
    'surprised':'3', #惊讶
    'normal':'4', #中性
}

if __name__ == '__main__':
    dir_path = os.path.join(os.getcwd(),'masked_imgs')
    # df = pd.DataFrame(columns=['No','emotion','pixels','Usage'],dtype=int)
    # df.set_index(['No'], inplace=True)
    dataset = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        print(root)
        widgets = ['Progress: ',progressbar.Percentage(), ' ', progressbar.Bar('#'),' ', progressbar.Timer(),' ', progressbar.ETA(),]
        p = progressbar.ProgressBar(widgets=widgets,maxval=len(files))
        p.start()
        i = 0
        for name in files:
            file_path = os.path.join(root, name)
            raw_info = str(file_path).split("\\")[-3:]
            if raw_info[1] in selected_emotions:
                img_array = np.asarray(cv2.imread(file_path, 0)).reshape(48*48)
                img_str = ' '.join(str(i) for i in img_array)
                info_dict = {'No':int(raw_info[2].replace('.jpg','')),'emotion':int(selected_emotions[raw_info[1]]), 'pixels':img_str, 'Usage':raw_info[0]}
                # df.append(info_dict,ignore_index=True)
                dataset.append(info_dict)
            i+=1
            p.update(i)
        p.finish()
    df = pd.DataFrame(dataset,columns=['No','emotion','pixels','Usage'],dtype=int)
    df.set_index(['No'], inplace=True)
    df.sort_index(inplace=True)
    df.to_csv('fer2013_masked_5_last.csv', index=False)
    # print(df)

            