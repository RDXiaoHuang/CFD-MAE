import torch
import torch.nn as nn
import numpy as np
from ..Common import Concat, CSP, Conv


class Neck(nn.Module):
    def __init__(self, width, depth, csp, pretrained=None):
        super().__init__()
        
        self.up_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_h1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.concat = Concat()

        self.h1 = CSP(width[4] + width[5], width[4], depth[0], csp[0])
        self.h2 = CSP(width[4] + width[4], width[3], depth[0], csp[0])
        
        # Standard Conv for downsampling
        self.h3 = Conv(width[3], width[3], 3, 2, 1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], csp[0])
        
        self.h5 = Conv(width[4], width[4], 3, 2, 1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], csp[1])
        
        if pretrained:
            self.load_weights(pretrained)
    
    def load_weights(self, pretrained_path):
        print(f"Loading neck weights from {pretrained_path}")
        try:
            # First try with weights_only=True
            try:
                pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fallback: register the module path that the weights file expects
                import sys
                from nets import nn as nets_nn_module
                sys.modules['nets.nn'] = nets_nn_module
                
                pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            if 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            if hasattr(pretrained_dict, 'state_dict'):
                pretrained_dict = pretrained_dict.state_dict()
            
            model_dict = self.state_dict()
            load_key, no_load_key, temp_dict = [], [], {}
            
            for k, v in pretrained_dict.items():
                k = k.replace('module.', '')
                
                # Map keys from standard YOLOv11 neck/head structure
                # YOLOv11 uses 'neck' or sometimes 'head' for this part
                if k.startswith('neck.'):
                    k = k.replace('neck.', '')
                elif k.startswith('head.'):
                    # Some YOLOv11 versions call it 'head'
                    k = k.replace('head.', '')
                
                # Map specific layer names if needed
                # Adjust these mappings based on actual YOLOv11 structure
                if k.startswith('n1.'): k = k.replace('n1.', 'h1.')
                elif k.startswith('n2.'): k = k.replace('n2.', 'h2.')
                elif k.startswith('n3.'): k = k.replace('n3.', 'h3.')
                elif k.startswith('n4.'): k = k.replace('n4.', 'h4.')
                elif k.startswith('n5.'): k = k.replace('n5.', 'h5.')
                elif k.startswith('n6.'): k = k.replace('n6.', 'h6.')
                
                if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            
            model_dict.update(temp_dict)
            self.load_state_dict(model_dict)
            print(f"Neck loaded {len(load_key)} layers")
        except Exception as e:
            print(f"Failed to load neck weights: {e}")

    def forward(self, x):
        p3, p4, p5 = x
        
        # Top-down path
        p5_up = self.up_p5(p5)
        h1 = self.h1(self.concat([p5_up, p4]))
        
        h1_up = self.up_h1(h1)
        h2 = self.h2(self.concat([h1_up, p3]))

        # Bottom-up path
        h3 = self.h3(h2)
        h4 = self.h4(self.concat([h3, h1]))
        h6 = self.h6(self.concat([self.h5(h4), p5]))
        
        return h2, h4, h6