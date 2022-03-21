from collections import OrderedDict
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import hiddenlayer as hl
import matplotlib.pyplot as plt
from torchsummary import summary
torch.cuda.empty_cache()

#enc1 torch.Size([2, 32, 512, 512]) enc3 torch.Size([2, 128, 128, 128]) 
#dec3 torch.Size([2, 128, 128, 128]) dec1 torch.Size([2, 32, 512, 512])
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.disc1_conv=nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=(1,1))
        self.inst1_norm=nn.InstanceNorm2d(32)
        self.leaky1_relu=nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        
        self.disc2_conv=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(1,1))
        self.inst2_norm=nn.InstanceNorm2d(128)
        self.leaky2_relu=nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        
        
        self.disc3_conv=nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=(1,1))
        self.inst3_norm=nn.InstanceNorm2d(128)
        self.leaky3_relu=nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        
        self.disc4_conv=nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=(1,1))
        self.inst4_norm=nn.InstanceNorm2d(32)
        self.leaky4_relu=nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        
        self.disc5_conv=nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=(1,1))
        self.inst5_norm=nn.InstanceNorm2d(32)
        self.leaky5_relu=nn.LeakyReLU(negative_slope=0.2, inplace=True)


        self.end_conv1=nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=(1,1))
        



                        
    def forward(self,model_input,out_segmap,feature_0,feature_1,feature_2,feature_3):
            
        #CAT_1
        cat1=torch.cat((out_segmap, model_input), dim=1)

        
        #DISC_1
        x0=self.disc1_conv(cat1)
        x0=self.inst1_norm(x0)
        x0=self.leaky1_relu(x0)



        #CAT_2            
        feature_0 = F.interpolate(feature_0, size=x0.shape[2])
        cat2=torch.cat((x0, feature_0), dim=1)

        
        #DISC_2
        x1=self.disc2_conv(cat2)
        x1=self.inst2_norm(x1)
        x1=self.leaky2_relu(x1)
        
        
        #CAT_3          
        feature_1 = F.interpolate(feature_1, size=x0.shape[2])
        cat3=torch.cat((x1, feature_1), dim=1)

        
        #DISC_3
        x2=self.disc3_conv(cat3)
        x2=self.inst3_norm(x2)
        x2=self.leaky3_relu(x2)
        
        
        #CAT_4        
        feature_2 = F.interpolate(feature_2, size=x0.shape[2])
        cat4=torch.cat((x2, feature_2), dim=1)

        
        #DISC_3
        x3=self.disc4_conv(cat4)
        x3=self.inst4_norm(x3)
        x3=self.leaky4_relu(x3)
        
        
        #CAT_5      
        feature_3 = F.interpolate(feature_3, size=x0.shape[2])
        cat5=torch.cat((x3, feature_3), dim=1)

        
        #DISC_4
        x4=self.disc5_conv(cat5)
        x4=self.inst5_norm(x4)
        x4=self.leaky5_relu(x4)

        x5=self.end_conv1(x4)

        
        return x5
 
