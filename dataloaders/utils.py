import math 
import numpy as np

# necessary functions
from scipy.fftpack import fft,fftfreq,ifft
# Importing Scipy 
import scipy as sp
import pywt
import scipy.fft as F
from sklearn.cluster import KMeans
import torch


def mag_3_signals(x,y,z):# magnitude function redefintion
    return np.array([math.sqrt((x[i]**2+y[i]**2+z[i]**2)) for i in range(len(x))])

def components_selection_one_signal(t_signal,freq1,freq2,sampling_freq):
    """
    DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz] 
                                                                (-0.3 and 0.3 are included)
    
    noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz] 
                                                                  (-25 and 25 hz inculded 20hz and -20hz not included)
    
    selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz] 
                                                                  (-0.3 and 0.3 not included , -20hz and 20 hz included)
    """

    t_signal=np.array(t_signal)
    t_signal_length=len(t_signal) # number of points in a t_signal
    
    # the t_signal in frequency domain after applying fft
    f_signal=fft(t_signal) # 1D numpy array contains complex values (in C)
    
    # generate frequencies associated to f_signal complex values
    freqs=np.array(sp.fftpack.fftfreq(t_signal_length, d=1/float(sampling_freq))) # frequency values between [-25hz:+25hz]
    

    
    
    f_DC_signal=[] # DC_component in freq domain
    f_body_signal=[] # body component in freq domain numpy.append(a, a[0])
    f_noise_signal=[] # noise in freq domain
    
    for i in range(len(freqs)):# iterate over all available frequencies
        
        # selecting the frequency value
        freq=freqs[i]
        
        # selecting the f_signal value associated to freq
        value= f_signal[i]
        
        # Selecting DC_component values 
        if abs(freq)>freq1:# testing if freq is outside DC_component frequency ranges
            f_DC_signal.append(float(0)) # add 0 to  the  list if it was the case (the value should not be added)                                       
        else: # if freq is inside DC_component frequency ranges 
            f_DC_signal.append(value) # add f_signal value to f_DC_signal list
    
        # Selecting noise component values 
        if (abs(freq)<=freq2):# testing if freq is outside noise frequency ranges 
            f_noise_signal.append(float(0)) # # add 0 to  f_noise_signal list if it was the case 
        else:# if freq is inside noise frequency ranges 
            f_noise_signal.append(value) # add f_signal value to f_noise_signal

        # Selecting body_component values 
        if (abs(freq)<=freq1 or abs(freq)>freq2):# testing if freq is outside Body_component frequency ranges
            f_body_signal.append(float(0))# add 0 to  f_body_signal list
        else:# if freq is inside Body_component frequency ranges
            f_body_signal.append(value) # add f_signal value to f_body_signal list
    
    ################### Inverse the transformation of signals in freq domain ########################
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    t_DC_component= ifft(np.array(f_DC_signal)).real
    t_body_component= ifft(np.array(f_body_signal)).real
    #t_noise=ifft(np.array(f_noise_signal)).real
    
    #total_component=t_signal-t_noise # extracting the total component(filtered from noise) 
    #                                 #  by substracting noise from t_signal (the original signal).
    

    #return (total_component,t_DC_component,t_body_component,t_noise) 
    return (t_DC_component,t_body_component) 



class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        
    def fit(self, df):
        if self.norm_type == "standardization":
            self.mean = df.mean(0)
            self.std = df.std(0)
        elif self.norm_type == "minmax":
            self.max_val = df.max()
            self.min_val = df.min()
        elif self.norm_type == "per_sample_std":
            self.max_val = None
            self.min_val = None
        elif self.norm_type == "per_sample_minmax":
            self.max_val = None
            self.min_val = None
        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))



def PrepareWavelets(K, length=20, seed= 1):
    motherwavelets = []
    for family in pywt.families():
        for mother in pywt.wavelist(family):
            motherwavelets.append(mother)
    
    X = np.zeros([1,length])
    PSI = np.zeros([1,length])
    for mw_temp in motherwavelets:
        if mw_temp.startswith('gaus') or mw_temp.startswith('mexh') or mw_temp.startswith('morl') or mw_temp.startswith('cmor') or mw_temp.startswith('fbsp') or mw_temp.startswith('shan') or mw_temp.startswith('cgau'):
            pass
        else:
            param = pywt.Wavelet(mw_temp).wavefun(level=7)
            psi, x = param[1], param[-1]

            # normalization
            psi_sum = np.sum(psi)
            if np.abs(psi_sum) > 1:
                psi = psi / np.abs(psi_sum)
            x = x / max(x)

            # down sampling
            idx_ds = np.round(np.linspace(0, x.shape[0]-1, length)).astype(int)
            x = x[idx_ds]
            psi = psi[idx_ds]

            X = np.vstack((X, x.reshape(1,-1)))
            PSI = np.vstack((PSI, psi.reshape(1,-1)))

    X = X[1:,:]
    PSI = PSI[1:,:]

    # clustering
    FRE = np.zeros([1,length])
    for i in range(PSI.shape[0]):
        FRE = np.vstack((FRE, np.real(F.fft(PSI[i,:])).reshape(1,-1)))
    FRE = FRE[1:,:]

    PSI_extended = np.hstack((PSI, FRE))
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(PSI_extended)
    label = kmeans.labels_

    SelectedWavelet = np.zeros([1,length])
    for k in range(K):
        wavesidx = np.where(label==k)[0][0]
        SelectedWavelet = np.vstack((SelectedWavelet, PSI[wavesidx,:]))            

    return torch.tensor(SelectedWavelet[1:,:])


def FiltersExtention(Filters):
    K, WS = Filters.shape
    

    if WS%2==1:
        N_padding = int((WS-1)/2)
        N_ds = int(torch.log2(torch.tensor(WS-1)).floor()) - 2
    else:
        N_ds = int(torch.log2(torch.tensor(WS)).floor()) - 2
        N_padding = int(WS/2)

    Filter_temp = Filters.repeat(N_ds,1,1)
    m = torch.nn.ConstantPad1d(N_padding, 0)
    
    for n_ds in range(N_ds-1):
        filter_temp = Filter_temp[n_ds,:,:]
        filter_temp_pad = m(filter_temp)
        filter_ds = filter_temp_pad[:,::2] * 2.
        Filter_temp[n_ds+1,:,:] = filter_ds
    
    # formualte dimensionality
    Filter_temp = Filter_temp.view(K*N_ds,WS)
    Filter_temp = Filter_temp.repeat(1,1,1,1)
    Filter_temp = Filter_temp.permute(2,0,1,3)

    # normalization
    energy = torch.sum(Filter_temp, dim=3, keepdims=True)
    energy[torch.abs(energy)<=1] = 1.
    Filter_temp = Filter_temp / energy

    return Filter_temp










