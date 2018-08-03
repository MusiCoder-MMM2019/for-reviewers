import os
import json
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import pylab
import scipy
from PIL import Image
import librosa
import librosa.display
import numpy as np
import multiprocessing as mp
from pydub import AudioSegment


class encoder_decoder:

    def __init__(self,audio_path):

        self.limit=-80 
        self.max_volume=256 
        self.mode_pace={'LOFI':10,'STD':15,'HIFI':25}

        self.base_name=self.getBaseName(audio_path) 
        self.content_auido_path=audio_path
        self.trans_audio_path  ='./temp/'+self.base_name+'_raw.wav'
        self.content_img_path  ='./temp/'+self.base_name+'.jpg'
        self.trans_img_path    ='./temp/'+self.base_name+'_trans.jpg'
        print(self.trans_img_path)

    def getBaseName(self,audio_path):
        base_name=os.path.splitext(os.path.basename(audio_path))[0]
        return base_name
    
    def setTransImgPath(self,texture):
        self.trans_img_path   ='./temp/'+self.base_name+'_'+texture+'.jpg'
        pass

    def setTransSoundPath(self,texture,quality):
        self.trans_audio_path ='./temp/'+self.base_name+'_'+texture+'_'+quality+'.mp3'
        pass

    def getSpectrumMatrix(self,audio_path):
        print(audio_path)
        sig, fs = librosa.load(audio_path,sr=None)
        D = np.abs(librosa.stft(sig))
        self.max_volume=np.max(D)
        D=librosa.amplitude_to_db(D,ref=np.max)
        return D

    def audio2img(self): 
        spectrum = self.getSpectrumMatrix(self.content_auido_path)
        self.limit=np.min(spectrum) 
        spectrum[spectrum<0.618*self.limit]=self.limit 

        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        librosa.display.specshow(spectrum,cmap='magma',x_axis='time')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10.25, 8.62)
        fig.savefig(self.content_img_path, dpi=100) 

        fig.clf()

        return True

    def readMagmaDiff(self): 
        magma_dif_list=[]
        with open('./data/magma_diff.json','r') as magma_file:
            magma_dif_list = json.loads(magma_file.read())['_magma_difference']
        return magma_dif_list

    def readImg(self,spectrum_path): 
        img_file=Image.open(spectrum_path)
        img_file = img_file.resize((862, 1025),Image.ANTIALIAS) 
        mtx_rgb=np.array(img_file.transpose(Image.FLIP_TOP_BOTTOM))
        mtx_rgb_sum=np.sum(mtx_rgb,2)/float(255)
        return mtx_rgb_sum

    def curver(self,mtx_rgb_sum,magma_dif_list): 
        mtx_value=np.ones(mtx_rgb_sum.shape)
        mtx_unit=np.ones(mtx_rgb_sum.shape) 
        i=0
        for dif in magma_dif_list:
            mtx_rgb_sum=mtx_rgb_sum-dif*mtx_unit 
            mtx_value[mtx_rgb_sum<0]=i/float(255)
            mtx_rgb_sum[mtx_rgb_sum<0]=3 
            i=i+1
        return mtx_value

    def GLA(self,index,mtx_amp,pace): 
        phase = np.exp(1.j * np.random.uniform(0., 2*np.pi, size=mtx_amp.shape))
        x_ = librosa.istft(mtx_amp * phase)
        for i in range(pace*10):
            _, phase = librosa.magphase(librosa.stft(x_))
            x_ = librosa.istft(mtx_amp * phase)
        return x_

    def wav2mp3(self,filepath):
        filedir,basename=os.path.split(filepath)
        barename=os.path.splitext(basename)[0]
        filepath_new=os.path.join(filedir,barename+'.mp3')
        AudioSegment.from_wav(filepath).export(filepath_new)
        return filepath_new

    def reconstructer(self,mtx_amp,mode):
        pace=self.mode_pace.get(mode,self.mode_pace.get('STD'))
        mtx_amp=(self.max_volume/np.max(mtx_amp))*mtx_amp
        mtx_amp_segs = [[0,mtx_amp[:,0*173:1*173],pace],
                        [1,mtx_amp[:,1*173:2*173],pace],
                        [2,mtx_amp[:,2*173:3*173],pace],
                        [3,mtx_amp[:,3*173:4*173],pace],
                        [4,mtx_amp[:,4*173:5*173],pace]]
        try:
            pool=mp.Pool()
            seg_songs=pool.starmap(self.GLA,mtx_amp_segs) 
        finally:
            pool.close()
            pool.join() 
        combined_sounds=np.hstack((seg_songs[0],seg_songs[1],seg_songs[2],seg_songs[3],seg_songs[4]))
 
        scipy.io.wavfile.write(self.trans_audio_path, 44100, combined_sounds) 
      
        self.trans_audio_path=(self.wav2mp3(self.trans_audio_path)).strip('\n')
        print(self.trans_audio_path)
        repr(self.trans_audio_path)
        return True

    def img2audio(self,mode): 
        magma_dif_list=self.readMagmaDiff()
        mtx_rgb_sum=self.readImg(self.trans_img_path)
        mtx_value=self.curver(mtx_rgb_sum,magma_dif_list)

        mtx_unit=np.ones(mtx_value.shape)
        mtx_db=(mtx_value/np.max(mtx_value)-mtx_unit)*float(abs(self.limit))
        mtx_amp=librosa.db_to_amplitude(mtx_db,ref=1.0)

        return self.reconstructer(mtx_amp,mode)

class converter:
    def __init__(self,base_name):
        self.model_paths={
                        'water':'converter/models/water.ckpt',
                        'future':'converter/models/future.ckpt',
                        'laser':'converter/models/laser.ckpt'

        }
        self.base_name=base_name

    def run(self,orig_img_path,trans_img_path,texture):
        result=os.popen('python3 converter/run.py '+\
                        '--content '+orig_img_path+' '+\
                        '--texture_model '+self.model_paths.get(texture,'converter/models/water.ckpt')+' '+\
                        '--output '+trans_img_path+' &').read()
        if "DONE" in result:
            return True
        else:
            print(result)
            return False
