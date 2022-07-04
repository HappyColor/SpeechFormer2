
import fairseq
import soundfile as sf
import scipy.signal as signal
from scipy import io
import torch
import torch.nn.functional as F
import os

class Wav2vec2(object):
    def __init__(self, ckpt_path, max_chunk=1600000):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.max_chunk = max_chunk

    def read_audio(self, path):
        wav, sr = sf.read(path)
        
        if sr != self.task.cfg.sample_rate:
            num = int((wav.shape[0]) / sr * self.task.cfg.sample_rate)
            wav = signal.resample(wav, num)
            print(f'Resample {sr} to {self.task.cfg.sample_rate}')
        
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim

        return wav

    def get_feats(self, path, layer):
        '''Layer index starts from 0. (e.g. 0-23)
        '''
        x = self.read_audio(path)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk = self.model.extract_features(source=x_chunk, padding_mask=None, mask=False, layer=layer-1)

                # feat_chunk is a dict. Keys: ['x', 'padding_mask', 'features', 'layer_results']
                # feat_chunk['x']: Output from transformer stack (target layer / last layer). Shape: B, T, 1024
                # feat_chunk['features']: Output from CNN extractor. Shape: B, T, 512
                # feat_chunk['layer_results'][i][0]: Output from i-th layer. Shape: T, B, 1024
                # Note: feat_chunk['x'] == feat_chunk['layer_results'][-1][0].transpose(0, 1)
                
                feat.append(feat_chunk['x'])
                
        return torch.cat(feat, 1).squeeze(0)

def extract_w2v2(model: Wav2vec2, layer, wavfile, savefile):
    fea = model.get_feats(wavfile, layer=layer)

    fea = fea.cpu().detach().numpy()   # (t, 1024)
    dict = {'w2v2': fea}
    io.savemat(savefile, dict)
    
    print(savefile, '->', fea.shape)

def handle_iemocap(model: Wav2vec2):
    matroot = '/148Dataset/data-chen.weidong/iemocap/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/iemocap/feature/w2v2_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/iemocap/feature/w2v2_large_L24_mat'

    if not os.path.exists(save_L12):
        os.makedirs(save_L12)
    if not os.path.exists(save_L24):
        os.makedirs(save_L24)

    mats = os.listdir(matroot)
    print(f'We have {len(mats)} samples in total.')
    for mat in mats:
        ses = mat[4]
        folder = mat[:-5]
        wavfile = f'/148Dataset/data-chen.weidong/iemocap/Session{ses}/sentences/wav/{folder}/{mat}.wav'
        savefile_L12 = os.path.join(save_L12, mat)
        savefile_L24 = os.path.join(save_L24, mat)
        extract_w2v2(model, 12, wavfile, savefile_L12)
        extract_w2v2(model, 24, wavfile, savefile_L24)

def handle_meld(model: Wav2vec2):
    matroot = '/148Dataset/data-chen.weidong/meld/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/meld/feature/w2v2_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/meld/feature/w2v2_large_L24_mat'

    state = ['train', 'dev', 'test']
    for s in state:
        matroot_s = os.path.join(matroot, s)
        save_L12_s = os.path.join(save_L12, s)
        save_L24_s = os.path.join(save_L24, s)

        if not os.path.exists(save_L12_s):
            os.makedirs(save_L12_s)
        if not os.path.exists(save_L24_s):
            os.makedirs(save_L24_s)

        mats = os.listdir(matroot_s)
        print(f'We have {len(mats)} samples in total.')
        for mat in mats:
            wavfile = f'/148Dataset/data-chen.weidong/meld/audio/{s}/{mat}.wav'
            savefile_L12 = os.path.join(save_L12_s, mat)
            savefile_L24 = os.path.join(save_L24_s, mat)
            extract_w2v2(model, 12, wavfile, savefile_L12)
            extract_w2v2(model, 24, wavfile, savefile_L24)

def handle_pitt(model: Wav2vec2):
    matroot = '/148Dataset/data-chen.weidong/DementiaBank/Pitt/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/DementiaBank/Pitt/feature/w2v2_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/DementiaBank/Pitt/feature/w2v2_large_L24_mat'

    state = ['Control', 'Dementia']
    for s in state:
        matroot_s = os.path.join(matroot, s, 'cookie')
        save_L12_s = os.path.join(save_L12, s, 'cookie')
        save_L24_s = os.path.join(save_L24, s, 'cookie')

        if not os.path.exists(save_L12_s):
            os.makedirs(save_L12_s)
        if not os.path.exists(save_L24_s):
            os.makedirs(save_L24_s)

        mats = os.listdir(matroot_s)
        print(f'We have {len(mats)} samples in total.')
        for mat in mats:
            wavfile = f'/148Dataset/data-chen.weidong/DementiaBank/Pitt/audio/utterance_wav/{s}/cookie/{mat}.wav'
            savefile_L12 = os.path.join(save_L12_s, mat)
            savefile_L24 = os.path.join(save_L24_s, mat)
            extract_w2v2(model, 12, wavfile, savefile_L12)
            extract_w2v2(model, 24, wavfile, savefile_L24)

def handle_daic(model: Wav2vec2):
    matroot = '/148Dataset/data-chen.weidong/AVEC2017/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/AVEC2017/feature/w2v2_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/AVEC2017/feature/w2v2_large_L24_mat'

    if not os.path.exists(save_L12):
        os.makedirs(save_L12)
    if not os.path.exists(save_L24):
        os.makedirs(save_L24)

    mats = os.listdir(matroot)
    print(f'We have {len(mats)} samples in total.')
    for mat in mats:
        wavfile = f'/148Dataset/data-chen.weidong/AVEC2017/audio/separate_wav/{mat}_AUDIO.wav'
        savefile_L12 = os.path.join(save_L12, mat)
        savefile_L24 = os.path.join(save_L24, mat)
        extract_w2v2(model, 12, wavfile, savefile_L12)
        extract_w2v2(model, 24, wavfile, savefile_L24)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    ckpt_path = "/148Dataset/data-chen.weidong/pre_trained_model/wav2vec-2.0/libri960_big.pt"
    model = Wav2vec2(ckpt_path)

    # handle_iemocap(model)
    # handle_meld(model)
    # handle_pitt(model)
    # handle_daic(model)
    
