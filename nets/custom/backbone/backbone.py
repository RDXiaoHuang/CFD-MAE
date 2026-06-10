import torch
import torch.nn as nn
import math
import numpy as np
from ..Common import Conv, CSP, SPP, PSA

class Backbone(nn.Module):
    def __init__(self, width, depth, csp, pretrained=None):
        super().__init__()
        
        self.width = width
        
        self.p1_conv = Conv(width[0], width[1], 3, 2)
        self.p2_conv = Conv(width[1], width[2], 3, 2)
        self.p3_conv = Conv(width[3], width[3], 3, 2)
        self.p4_conv = Conv(width[4], width[4], 3, 2)
        self.p5_conv = Conv(width[4], width[5], 3, 2)
        
        self.p2_csp = CSP(width[2], width[3], depth[0], csp[0], 4)
        self.p3_csp = CSP(width[3], width[4], depth[1], csp[0], 4)
        self.p4_csp = CSP(width[4], width[4], depth[2], csp[1])
        self.p5_csp = CSP(width[5], width[5], depth[3], csp[1])
        
        self.spp = SPP(width[5], 5)
        
        # Standard PSA
        self.psa = PSA(width[5], width[5], depth[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if pretrained:
            self.load_weights(pretrained)

    def load_weights(self, pretrained_path):
        print(f"Loading backbone weights from {pretrained_path}")
        try:
            # Try loading with weights_only=True first (safer)
            try:
                pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fallback: load full model and extract state_dict
                import sys
                from nets import nn as nets_nn_module
                
                # Register the module path that the weights file expects
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
                
                # Map keys from standard YOLOv11 or Darknet
                if k.startswith('backbone.p1.0.'): k = k.replace('backbone.p1.0.', 'p1_conv.')
                elif k.startswith('backbone.p2.0.'): k = k.replace('backbone.p2.0.', 'p2_conv.')
                elif k.startswith('backbone.p2.1.'): k = k.replace('backbone.p2.1.', 'p2_csp.')
                elif k.startswith('backbone.p3.0.'): k = k.replace('backbone.p3.0.', 'p3_conv.')
                elif k.startswith('backbone.p3.1.'): k = k.replace('backbone.p3.1.', 'p3_csp.')
                elif k.startswith('backbone.p4.0.'): k = k.replace('backbone.p4.0.', 'p4_conv.')
                elif k.startswith('backbone.p4.1.'): k = k.replace('backbone.p4.1.', 'p4_csp.')
                elif k.startswith('backbone.p5.0.'): k = k.replace('backbone.p5.0.', 'p5_conv.')
                elif k.startswith('backbone.p5.1.'): k = k.replace('backbone.p5.1.', 'p5_csp.')
                elif k.startswith('backbone.p5.2.'): k = k.replace('backbone.p5.2.', 'spp.')
                elif k.startswith('backbone.p5.3.'): k = k.replace('backbone.p5.3.', 'psa.')
                
                if k.startswith('backbone.'):
                    k = k.replace('backbone.', '')
                
                if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            
            model_dict.update(temp_dict)
            self.load_state_dict(model_dict)
            print(f"Backbone loaded {len(load_key)} layers")
            if no_load_key:
                print(f"Backbone skipped {len(no_load_key)} layers")
        except Exception as e:
            print(f"Failed to load backbone weights: {e}")

    def forward(self, x):
        p1 = self.p1_conv(x)
        p2 = self.p2_csp(self.p2_conv(p1))
        p3 = self.p3_csp(self.p3_conv(p2))
        p4 = self.p4_csp(self.p4_conv(p3))
        p5 = self.p5_conv(p4)
        p5 = self.p5_csp(p5)
        p5 = self.spp(p5)
        p5 = self.psa(p5)
        
        return p3, p4, p5

def Yolov11_Backbone(pretrained, **kwargs):
    return Backbone(pretrained=pretrained, **kwargs)
