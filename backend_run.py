import os
import backend

class task:
    
    def __init__(self,content_audio_path): 
        self.endecoder=backend.encoder_decoder(content_audio_path)
        self.converter=backend.converter(self.endecoder.base_name)

    def soundToImage(self): 
        return self.endecoder.audio2img()
   
    def convert(self,texture): 
        self.endecoder.setTransImgPath(texture)
        return self.converter.run(self.endecoder.content_img_path,self.endecoder.trans_img_path,texture)

    def imageToSound(self,quality,texture): 
        self.endecoder.setTransSoundPath(texture,quality)
        self.endecoder.img2audio(quality)

