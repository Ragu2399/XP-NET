import wandb
wandb.login()

user = "ragu2399"
project = "gan_trail"
display_name = "t2"

wandb.init(entity=user, project=project, name=display_name)

import torch
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from loss import *
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import Teacher_model
import student_model
import Discriminator_model
import os
from utils import (
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
torch.cuda.empty_cache()
random_seed = 242 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)
torch.cuda.is_available()


# Hyperparameters etc.

LEARNING_RATE = 1e-5
BETA_1=0.5
BETA_2=0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1000
NUM_WORKERS =12
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../ORDERED_ENDOCV_DATASET/IMAGES/"
TRAIN_MASK_DIR = "../ORDERED_ENDOCV_DATASET/MASKS/"
VAL_IMG_DIR = "../ORDERED_ENDOCV_DATASET/VALIDATION_IMAGES/"
VAL_MASK_DIR = "../ORDERED_ENDOCV_DATASET/VALIDATION_MASK/"
CLIP_LIMIT=6
BATCH_INDEX=400
NUM_IMAGES=os.listdir(TRAIN_IMG_DIR)
NUM_IMAGES=len(NUM_IMAGES)
print(f'Total number of Images : {NUM_IMAGES}')

TEACHER_1_PATH="./WEIGHTS/TEACHER_1.pth" #SPECIFY TEACHER 1 MODEL PATH
TEACHER_2_PATH="./WEIGHTS/TEACHER_2.pth" #SPECIFY TEACHER 2 MODEL PATH
STUDENT_PATH="./WEIGHTS/STUDENT.pth" #SPECIFY STUDENT MODEL PATH
DISCRIMINATOR_PATH="./WEIGHTS/DISCRIMINATOR.pth" #SPECIFY TEACHER 1 MODEL PATH


#ALL_LOSS_FUNCTIONS
lossfn_1 = DiceLoss()
lossfn_2 = DiceBCELoss()
lossfn_3 = IoULoss()
lossfn_4 = FocalLoss()
lossfn_5 = TverskyLoss()
lossfn_6 = FocalTverskyLoss()
lossfn_7 = ComboLoss()


def get_models(DEVICE):  
    
    # Teacher network. Note: For BraTS, the total number of modalities/sequences is 4. Also, BraTS has 4 segmentation classes.
    teacher_model =Teacher_model.UNet_SAB().to(DEVICE)
    teacher_pretrained_statedict = torch.load(TEACHER_PATH)
    teacher_model.load_state_dict(teacher_pretrained_statedict["state_dict"])
    print("Created teacher model and with trained_weights.")

    # Freeze teacher weights.
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    # Student network. Note: For BraTS, there is only 1 post-contrast modalities/sequences, so the student recieves 3. 
    # Also, BraTS has 4 segmentation classes.
    generator_model = student_model.UNet_SAB_STUDENT().to(DEVICE)
    generator_pretrained_statedict = torch.load(STUDENT_PATH)
    generator_model.load_state_dict(generator_pretrained_statedict["state_dict"])
    print(f' Created generator model (i.e., the student model) from {STUDENT_PATH}')

    # Discriminator network. The discriminator recieves the student input modalities/sequences (i.e., 3 for BraTS) and the 
    # output segmentation map (i.e., 4 classes for BraTS) as input (i.e., a total of 3+4 channels).
    discriminator_model = Discriminator_model.Discriminator().to(DEVICE)
    print("Created discriminator/critic.")
    
    return teacher_model, generator_model, discriminator_model



def get_optimizers(generator_model, discriminator_model):

    # Select an optimizer for the generator
    gen_optimizer = torch.optim.Adam(generator_model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

    # Select an optimizer for the discriminator
    disc_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=LEARNING_RATE, betas=(BETA_1,BETA_2))
    
    return gen_optimizer, disc_optimizer



def get_criteria(device):
    
    # Criterion
    criterion_MSE = torch.nn.MSELoss().to(device)

    criterion_CE = nn.CrossEntropyLoss().to(device)

    # softmax
    n_softmax = nn.Softmax(dim=1).to(device)
    
    return criterion_MSE, criterion_CE, n_softmax



def load_pre_trained_model(generator_model,discriminator_model):


    # LOAD PRE-TRINAED STUDENT
    student_pretrained_statedict = torch.load(STUDENT_PATH)
    generator_model.load_state_dict(student_pretrained_statedict["state_dict"])
    print(f' Generator model loaded from {STUDENT_PATH}' )

    # LOAD PRE-TRINAED DISCRIMINATOR
    Discriminator_pretrained_statedict = torch.load(DISCRIMINATOR_PATH)
    discriminator_model.load_state_dict(Discriminator_pretrained_statedict["state_dict"])
    print(f' DISC model loaded from {DISCRIMINATOR_PATH}' )

        
    return generator_model,discriminator_model 

def Plot(input_image,predicted,target):
    input_image=input_image.cpu().detach().numpy().astype(np.float32)
    predicted=predicted.cpu().detach().numpy().astype(np.float32)
    target=target.cpu().detach().numpy().astype(np.float32)
    
    num_images=input_image.shape[0]
    
    for image,pred,targ in zip(input_image,predicted,target):

        
        image=np.swapaxes(image,0,2)
        pred=np.swapaxes(pred,0,2)
        targ=np.swapaxes(targ,0,2)

        
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1,title='input')
        plt.imshow(image)
        plt.subplot(1,3,2,title='predicted')
        pred[pred >= 0.5]=1.0
        pred[pred <  0.5]=0.0
        plt.imshow(pred)
        plt.subplot(1,3,3,title='Target')
        plt.imshow(targ)
        plt.show()

def train_log(loss_1,batch_idx,epoch):
    wandb.log({"DiceLoss": loss_1},step=batch_idx)
    
    
def training_epoch(models, criteria, optimizers, trainset, device,epoch):
    
    # Unpack
    teacher_model, generator_model, discriminator_model = models
    criterion_MSE, criterion_CE  = criteria
    gen_optimizer, disc_optimizer = optimizers
    
    # Counter
    sample_count = 0.0
    
    # Loss tracker
    mean_LS = 0.0
    mean_LHD = 0.0

    # Set to train
    generator_model.train()
    teacher_model.train()
    discriminator_model.train()

    
    loop = tqdm(trainset)
    # Go over each batch of the training set

    for batch_idx, (datas, targets) in enumerate(loop):
#     datas,targets=next(iter(loop))

#         print("--- SAMPLE",int(sample_count)," ---")

        ###############################################
        ############### GET INPUT DATA ################
        ###############################################        

        x_teacher = datas.to(device=DEVICE)
        x_student = datas.to(device=DEVICE)
        y = targets.long().unsqueeze(1).to(device=DEVICE)


        # Get the teacher output for this sample. This will be the real segmentation map.
        with torch.no_grad():
            real_segmap, real_features_0, real_features_1,real_features_2,real_features_3 = teacher_model(x_teacher)

        ###########################################
        ########### TRAIN THE GENERATOR ###########
        ###########################################

        # zero the generator gradient
        gen_optimizer.zero_grad()

        # Run the input data through the generator
        fake_segmap, fake_features_0, fake_features_1, fake_features_2, fake_features_3 = generator_model(x_student)

        # Feed the disc the "fake" data
        disc_fake_adv = discriminator_model(fake_segmap, x_student,  fake_features_0, fake_features_1, fake_features_2, fake_features_3)

        # Create real and fake labels.
        disc_out_shape = disc_fake_adv.shape
        real = torch.ones(disc_out_shape).to(device)
        fake = torch.zeros(disc_out_shape).to(device)

        # Compute adversarial loss.
        gen_loss_GAN = criterion_MSE(disc_fake_adv, real)

        # Compute voxel-base loss with GROUND TRUTH
        gen_loss_VOX=lossfn_1(fake_segmap, y)

    #         gen_loss_VOX = criterion_CE(fake_segmap, y)

        # Compute TOTAL gen loss, back-propogate, and step generator optimizer forward
        gen_loss = gen_loss_VOX + ((0.2)*gen_loss_GAN)
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator_model.parameters(), CLIP_LIMIT)
        gen_optimizer.step()


        #PLOT GENERATOR OUTPUT
    #     if batch_idx%BATCH_INDEX==0:
        if batch_idx%BATCH_INDEX==0:

            print(f'EPOCH : {epoch}')
            print(f'DICE loss  : {gen_loss_VOX}')

            Plot(x_teacher,fake_segmap,y)
            if epoch==0:
                train_log(gen_loss_VOX,(batch_idx*BATCH_SIZE),epoch)
            else:
                train_log(gen_loss_VOX,((batch_idx*BATCH_SIZE)+(NUM_IMAGES*epoch)),epoch)

        ###############################################
        ########### TRAIN THE DISCRIMINATOR ###########
        ###############################################

        # zero the discriminator gradient
        disc_optimizer.zero_grad()

        # REMOVE GRADIENT FOR GENERATOR
        fake_segmap = fake_segmap.detach()
        fake_features_0=fake_features_0.detach()
        fake_features_1=fake_features_1.detach()
        fake_features_2=fake_features_2.detach()
        fake_features_3=fake_features_3.detach()


        # Feed the disc the "real" data                    r
        disc_real = discriminator_model(real_segmap, x_student, real_features_0, real_features_1,real_features_2,real_features_3)
        disc_loss_real = criterion_MSE(disc_real, real)


        # Feed the disc the "fake" data 
        disc_fake = discriminator_model(fake_segmap, x_student,  fake_features_0, fake_features_1, fake_features_2, fake_features_3 )
        disc_loss_fake = criterion_MSE(disc_fake, fake)


        # Compute total discriminator loss.
        disc_loss = disc_loss_real + disc_loss_fake

        # Determine if we should update the discriminator for this sample.
        disc_real_mean = torch.mean(torch.ge(disc_real,0.5).float())
        disc_fake_mean = torch.mean(torch.le(disc_fake,0.5).float())
        disc_mean = (disc_real_mean + disc_fake_mean)/2.0

        # Back-propogate the loss and step discriminator optimizer forward, if discriminator performance is
        # under the threshold.
        if(disc_mean <= 0.8):
            disc_loss.backward()    
            disc_optimizer.step()


        # Move data from GPU to CPU. This is done in order to prevent a strange CUDA error encountered during training, which 
        # prints the message: "CUDA: an illegal memory access was encountered".
        x_teacher = x_teacher.detach().to('cpu')
        x_student = x_student.detach().to('cpu')
        y = y.detach().to('cpu')
        real_segmap = real_segmap.detach().to('cpu')
        real_features_0=real_features_0.detach()
        real_features_1=real_features_1.detach()
        real_features_2=real_features_2.detach()
        real_features_3=real_features_3.detach()

        disc_fake_adv = disc_fake_adv.detach().to('cpu')
        real = real.detach().to('cpu')
        fake = fake.detach().to('cpu')
        gen_loss_GAN = gen_loss_GAN.detach().to('cpu')
        gen_loss_VOX = gen_loss_VOX.detach().to('cpu')
        gen_loss = gen_loss.detach().to('cpu')
        fake_segmap = fake_segmap.detach().to('cpu')
        fake_features_0=fake_features_0.detach()
        fake_features_1=fake_features_1.detach()
        fake_features_2=fake_features_2.detach()
        fake_features_3=fake_features_3.detach()

        disc_real = disc_real.detach().to('cpu')
        disc_loss_real = disc_loss_real.detach().to('cpu')
        disc_fake = disc_fake.detach().to('cpu')
        disc_loss_fake = disc_loss_fake.detach().to('cpu')
        disc_loss = disc_loss.detach().to('cpu')
        disc_real_mean = disc_real_mean.detach().to('cpu')
        disc_fake_mean = disc_fake_mean.detach().to('cpu')
        disc_mean = disc_mean.detach().to('cpu')

        # Update loss trackers.
        mean_LS = mean_LS + gen_loss.item()
        mean_LHD = mean_LHD + disc_loss.item()

        # Increment sample counter. 
        sample_count+=1.0

        # Find epoch loss.
        mean_LS = mean_LS/sample_count
        mean_LHD = mean_LHD/sample_count


        loop.set_postfix(loss=gen_loss_VOX.item())


    return (teacher_model, generator_model,discriminator_model), (gen_optimizer, disc_optimizer), (mean_LS, mean_LHD)


# Perform validation for this epoch.



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ],
    )


    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    validate_vm_list = []
    running_vm = 0.0
    #LOADING_MODEL
    teacher_model, generator_model, discriminator_model = get_models(DEVICE) 
    summary(teacher_model,(3,512,512))
    summary(generator_model,(3,512,512))
    summary(discriminator_model,[(1,512,512),(3,512,512),(32,512,512),(128,512,512),(128,512,512),(32,512,512)])
    #GETTING OPTIMIZERS
    gen_optimizer, disc_optimizer = get_optimizers(generator_model, discriminator_model)
    #GETTING_CRITERIA
    criterion_MSE, criterion_CE, n_softmax = get_criteria(DEVICE)
    
    
    #LOADING PRE TRAINED WEIGHTS
    if LOAD_MODEL:
        generator_model,discriminator_model = load_pre_trained_model( generator_model,discriminator_model)


    scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=10, gamma=0.1)


    for epoch in range(NUM_EPOCHS):
        print(f'{epoch} EPOCH   LR {scheduler.get_last_lr()}' )
        wandb.watch(generator_model,log="all")
        
        
        models, optimizers, training_losses = training_epoch((teacher_model, generator_model, discriminator_model), (criterion_MSE, criterion_CE), (gen_optimizer, disc_optimizer), train_loader, DEVICE,epoch)#         train_fn(train_loader, model, optimizer,epoch)#, loss_fn)#, scaler)


        teacher_model, generator_model, discriminator_model = models
        gen_optimizer, disc_optimizer = optimizers
        mean_LS, mean_LHD = training_losses      


        
        ###########################################
        ########## SAVE MODEL PARAMETERS ##########
        ###########################################



        ##Save generator and discriminator model and optimizer.
    
        gen_checkpoint = {
            "state_dict": generator_model.state_dict(),
            "optimizer":gen_optimizer.state_dict(),
        }
        model_name="Generator"
        save_checkpoint(gen_checkpoint,epoch,model_name)
        
        dis_checkpoint = {
            "state_dict": discriminator_model.state_dict(),
            "optimizer":disc_optimizer.state_dict(),
        }
        model_name="discriminator"
        save_checkpoint(dis_checkpoint,epoch,model_name)

        scheduler.step()



if __name__ == "__main__":
    main()
