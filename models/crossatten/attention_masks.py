
import torch
import math

####################             Mask                   #########################################

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.ones((L, L)).to(device)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
            
    @property   
    def mask(self):
        return self._mask

class LocalSymmetryMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            window_size = math.ceil(1.2*np.log2(L)/2)  #halb
            mask = torch.ones((L, L)).to(device)
            mask = torch.triu(mask,-window_size).T
            mask = torch.triu(mask,-window_size)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
    @property            
    def mask(self):
        return self._mask

class LocalLogSymmetryMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.zeros((L, L), dtype=torch.float).to(device)
            for i in range(L):
                mask[i] = self.row_mask(i, L)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)

            
    def row_mask(self,index, L):
        local_window_size = math.ceil(np.log2(L)/2) # 1/2 window size
        # 对当前行的index 行 进行初始化
        mask = torch.zeros((L), dtype=torch.float)

        if((index - local_window_size + 1) < 0):
            mask[:index] = 1 # Local attention
        else:
            mask[index - local_window_size + 1:(index + 1)] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index - local_window_size + 1 - int(1.5**i)
                if new_index >= 0:
                    mask[new_index] = 1
                else:
                    break
                    
        if ((index + local_window_size-1 )>=L):
            mask[index:] = 1 
        else:
            mask[index:index+local_window_size] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index + local_window_size-1 +int(1.5**i)
                if new_index < L:
                    mask[new_index] = 1
                else:
                    break
        return mask               

    @property          
    def mask(self):
        return self._mask

Mask_dict = {"Triangular"     :TriangularCausalMask,
             "LocalSymmetry"  :LocalSymmetryMask,
             "Full"           :FullMask,
             "LocLogSymmetry" :LocalLogSymmetryMask}