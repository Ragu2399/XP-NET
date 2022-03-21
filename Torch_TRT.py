import os
import sys
import common
import student_model
import numpy as np
import pycuda.autoinit
import tensorrt as trt
from matplotlib import pyplot as plt 
import h5py
from torchsummary import summary
np.set_printoptions(threshold=sys.maxsize)
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter, torch_dtype_to_trt
import datetime
torch.cuda.empty_cache()

MODEL_PATH="MODEL_4_GEN.pth"
MODEL= student_model.UNet_SAB_STUDENT()
checkpoint=torch.load(MODEL_PATH)
MODEL.load_state_dict(checkpoint["state_dict"])
MODEL.eval()
summary(MODEL.cuda(),(3,512,512))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    INPUT_NAME = "encoder1.conv1.weight"
    INPUT_SHAPE = (3, 512,512)
    OUTPUT_NAME = "SIGMOID"
    DTYPE = trt.float32
    
    
def addBatchNorm2d(network, weight_map, input_trt, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = (gamma / var).numpy()
    shift = (-mean / var * gamma + beta).numpy()

    batch_norm=network.add_scale(input= input_trt, mode=trt.ScaleMode.CHANNEL,shift=shift,scale=scale)
    
    return batch_norm



def Encoder_1(network,weights_map,input_tensor,EPS):
        
    #ENCODER_1_conv1
    encoder_1_conv1_weight = weights_map['encoder1.conv1.weight'].numpy()
    
    encoder_1_conv1 = network.add_convolution_nd(input=input_tensor, num_output_maps=32, kernel_shape=(3,3), kernel=encoder_1_conv1_weight)
    encoder_1_conv1.stride_nd = (1, 1)
    encoder_1_conv1.padding_nd = (1,1)
    assert encoder_1_conv1
    
    encoder_1_bn1 = addBatchNorm2d(network, weights_map, encoder_1_conv1.get_output(0),"encoder1.bn1" , EPS)
    assert encoder_1_bn1
    
    encoder_1_relu1 = network.add_activation(input=encoder_1_bn1.get_output(0), type=trt.ActivationType.RELU)
    assert encoder_1_relu1    
    
    
    #PSA
    PSA_out=PSA(network,weights_map,encoder_1_relu1.get_output(0),EPS)
    assert PSA_out 

        
    #ENCODER_1_conv2
    encoder_1_conv2_weight = weights_map['encoder1.conv2.weight'].numpy()
    
    encoder_1_conv2 = network.add_convolution_nd(input=PSA_out.get_output(0), num_output_maps=32, kernel_shape=(3,3), kernel=encoder_1_conv2_weight)
    encoder_1_conv2.stride_nd = (1, 1)
    encoder_1_conv2.padding_nd = (1,1)
    assert encoder_1_conv2
    
    encoder_1_bn2 = addBatchNorm2d(network, weights_map, encoder_1_conv2.get_output(0),"encoder1.bn2" , EPS)
    assert encoder_1_bn2
    
    encoder_1_relu2 = network.add_activation(input=encoder_1_bn2.get_output(0), type=trt.ActivationType.RELU)
    assert encoder_1_relu2 
    
    
    return encoder_1_relu2


def SE(network,weights_map,input_tensor,name,EPS):
    
    pooling_layer1=network.add_pooling_nd(input=input_tensor,type=trt.PoolingType.AVERAGE, window_size=(256,256))
    pooling_layer1.stride_nd = (256,256) 
    
    pooling_layer2=network.add_pooling_nd(input=pooling_layer1.get_output(0),type=trt.PoolingType.AVERAGE, window_size=(2,2))
    pooling_layer2.stride_nd= (1,1) 
    
    assert pooling_layer2

    sa_conv1_weights=weights_map[name +'.fc1.weight'].numpy()

    sa_conv1_bias=weights_map[name +'.fc1.bias'].numpy()
    
    sa_conv1 = network.add_convolution_nd(input=pooling_layer2.get_output(0), num_output_maps=1, kernel_shape=(1,1), kernel=sa_conv1_weights,bias=sa_conv1_bias)
    sa_conv1.padding_nd = (0,0)
    

    sa_conv1_relu1 = network.add_activation(input=sa_conv1.get_output(0), type=trt.ActivationType.RELU)
    assert sa_conv1_relu1  


    
    
    sa_conv2_weights=weights_map[name+'.fc2.weight'].numpy()
    sa_conv2_bias=weights_map[name+'.fc2.bias'].numpy()
    
    sa_conv2 = network.add_convolution_nd(input=sa_conv1_relu1.get_output(0), num_output_maps=8, kernel_shape=(1,1), kernel=sa_conv2_weights,bias=sa_conv2_bias)
    sa_conv2.padding_nd = (0,0)

    sa_conv2_sigmoid = network.add_activation(input=sa_conv2.get_output(0), type=trt.ActivationType.SIGMOID)
    assert sa_conv2_sigmoid  
    
    return sa_conv2_sigmoid


def PSA(network,weights_map,input_tensor,EPS):
    split_channel=8
    
    psa_conv1_weights=weights_map['encoder1.Psa.conv_1.weight'].numpy()

    psa_conv2_weights=weights_map['encoder1.Psa.conv_2.weight'].numpy()

    psa_conv3_weights=weights_map['encoder1.Psa.conv_3.weight'].numpy()

    psa_conv4_weights=weights_map['encoder1.Psa.conv_4.weight'].numpy()

    
    psa_conv1 = network.add_convolution_nd(input=input_tensor, num_output_maps=8, kernel_shape=(3,3), kernel=psa_conv1_weights)
    psa_conv1.stride_nd = (1, 1)
    psa_conv1.padding_nd = (1,1)
    psa_conv1.num_groups=1
    assert psa_conv1
    
    psa_conv2 = network.add_convolution_nd(input=input_tensor, num_output_maps=8, kernel_shape=(5,5), kernel=psa_conv2_weights)
    psa_conv2.stride_nd = (1, 1)
    psa_conv2.padding_nd = (2,2)
    psa_conv2.num_groups=2
    assert psa_conv2
    
    psa_conv3 = network.add_convolution_nd(input=input_tensor, num_output_maps=8, kernel_shape=(7,7), kernel=psa_conv3_weights)
    psa_conv3.stride_nd = (1, 1)
    psa_conv3.padding_nd = (3,3)
    psa_conv3.num_groups=4
    assert psa_conv3
    
    psa_conv4 = network.add_convolution_nd(input=input_tensor, num_output_maps=8, kernel_shape=(9,9), kernel=psa_conv4_weights)
    psa_conv4.stride_nd = (1, 1)
    psa_conv4.padding_nd = (4,4)
    psa_conv4.num_groups=8

    assert psa_conv4

    
    list_feat=[psa_conv4.get_output(0),psa_conv3.get_output(0),psa_conv2.get_output(0),psa_conv1.get_output(0)]
    feats=network.add_concatenation(list_feat)
    assert feats
    
    
    feats_reshape=network.add_shuffle(input=feats.get_output(0))
    feats_reshape.reshape_dims = [1, 4, split_channel, 512, 512]   

  #SE_BLOCK


    x1_se1_name="encoder1.Psa.se"
    x1_se1=SE(network,weights_map,psa_conv1.get_output(0),x1_se1_name,EPS)

    
    x2_se1_name="encoder1.Psa.se"
    x2_se1=SE(network,weights_map,psa_conv2.get_output(0),x1_se1_name,EPS)

    
    x3_se1_name="encoder1.Psa.se"
    x3_se1=SE(network,weights_map,psa_conv3.get_output(0),x1_se1_name,EPS)

    
    x4_se1_name="encoder1.Psa.se"
    x4_se1=SE(network,weights_map,psa_conv4.get_output(0),x1_se1_name,EPS)

    

    list_Xse_feat=[x4_se1.get_output(0),x3_se1.get_output(0),x2_se1.get_output(0),x1_se1.get_output(0)]
    xse_feats=network.add_concatenation(list_Xse_feat)
    assert xse_feats
    

    attention_vectors=network.add_shuffle(input=xse_feats.get_output(0))
    attention_vectors.reshape_dims = [1, 4, split_channel, 1,1]
    assert attention_vectors
    
    
    softmax_attention_vector = network.add_softmax(input=attention_vectors.get_output(0))
    assert softmax_attention_vector

    
    matmul = np.load('vectorsingle_mul.npy')
    feature_scalar = network.add_constant(trt.Dims((1, 4, 8, 1, 262144)), matmul)
    
    
    attetion_matrix = network.add_matrix_multiply(softmax_attention_vector.get_output(0), trt.MatrixOperation.NONE, feature_scalar.get_output(0), trt.MatrixOperation.NONE)
    assert attetion_matrix
    

    
    attetion_matrix_reshape=network.add_shuffle(input=attetion_matrix.get_output(0))
    attetion_matrix_reshape.reshape_dims = [1,4,split_channel, 512, 512]
    
    final_matrix = network.add_matrix_multiply(feats_reshape.get_output(0), trt.MatrixOperation.NONE, attetion_matrix_reshape.get_output(0), trt.MatrixOperation.NONE)
    assert final_matrix
    
    attetion_matrix_reshape2=network.add_shuffle(input=final_matrix.get_output(0))
    attetion_matrix_reshape2.reshape_dims = [1,32, 512, 512]
    


    return attetion_matrix_reshape2


def Encoder_2(network,weights_map,input_tensor,EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER):

    #ENCODER_1_conv1
    
    if DECODER:
        up_conv1_weight = weights_map["upconv"+str(dec_no)+".weight"].numpy()
        up_conv1_bias= weights_map["upconv"+str(dec_no)+".bias"].numpy()
        
        up_conv1 = network.add_deconvolution_nd(input=input_tensor, num_output_maps=features, kernel_shape=(2,2), kernel=up_conv1_weight,bias=up_conv1_bias)
        up_conv1.stride_nd = (2,2)
        
        
        skip_feat=[up_conv1.get_output(0),SKIP_LAYER]
        skip_feat_cat=network.add_concatenation(skip_feat)
        assert skip_feat_cat
        input_tensor=skip_feat_cat.get_output(0)
        assert input_tensor
        
#         print(f'input_tensor {input_tensor.shape}')

        
    encoder_2_conv1_weight = weights_map[layer_name+"P_conv1.weight"].numpy()
    encoder_2_conv1 = network.add_convolution_nd(input=input_tensor, num_output_maps=features, kernel_shape=(1,5), kernel=encoder_2_conv1_weight)
    encoder_2_conv1.padding_nd = (0,2)
    encoder_2_conv1.num_groups=16
    assert encoder_2_conv1
    
    
    
    encoder_2_conv2_weight = weights_map[layer_name+"D_conv1.weight"].numpy() 
    encoder_2_conv2 = network.add_convolution_nd(input=encoder_2_conv1.get_output(0), num_output_maps=features, kernel_shape=(1,1), kernel=encoder_2_conv2_weight)
    assert encoder_2_conv2
    
    encoder_2_bn1 = addBatchNorm2d(network, weights_map, encoder_2_conv2.get_output(0),layer_name+"norm1" , EPS)
    assert encoder_2_bn1
    
    encoder_2_relu1 = network.add_activation(input=encoder_2_bn1.get_output(0), type=trt.ActivationType.RELU)
    assert encoder_2_relu1    
    
    
    

    
    encoder_2_conv3_weight = weights_map[layer_name+"Pconv2.weight"].numpy()
    encoder_2_conv3 = network.add_convolution_nd(input=encoder_2_relu1.get_output(0), num_output_maps=features, kernel_shape=(5,1), kernel=encoder_2_conv3_weight)
    encoder_2_conv3.padding_nd = (2,0)
    encoder_2_conv3.num_groups=16
    assert encoder_2_conv3
    
    
    encoder_2_conv4_weight = weights_map[layer_name+"Dconv2.weight"].numpy() 
    encoder_2_conv4 = network.add_convolution_nd(input=encoder_2_conv3.get_output(0), num_output_maps=features, kernel_shape=(1,1), kernel=encoder_2_conv4_weight)
    assert encoder_2_conv4
    
    encoder_2_bn2 = addBatchNorm2d(network, weights_map, encoder_2_conv4.get_output(0),layer_name+"norm2" , EPS)
    assert encoder_2_bn2
    
    encoder_2_relu2 = network.add_activation(input=encoder_2_bn2.get_output(0), type=trt.ActivationType.RELU)
    assert encoder_2_relu2    
    
    
    if dec_no==1:
        end_conv1_weight = weights_map["conv.weight"].numpy()
        end_conv1_bias= weights_map["conv.bias"].numpy()
        
        end_conv1 = network.add_convolution_nd(input=encoder_2_relu2.get_output(0), num_output_maps=1, kernel_shape=(1,1), kernel=end_conv1_weight,bias=end_conv1_bias)
        
        end_sigmoid = network.add_activation(input=end_conv1.get_output(0), type=trt.ActivationType.SIGMOID)
        assert end_sigmoid  
        return end_sigmoid
    
    return encoder_2_relu2



def populate_network(network, weights_map):
    EPS = 1e-5
    # Configure the network layers based on the weights provided.
    
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    
    Encoder_1_out=Encoder_1(network,weights_map,input_tensor,EPS)
    
    Pooling_1=network.add_pooling_nd(input=Encoder_1_out.get_output(0),type=trt.PoolingType.MAX, window_size=trt.DimsHW(2, 2))
    Pooling_1.stride_nd=(2,2)
    assert Pooling_1

    features=64
    layer_name="encoder2.enc2"
    DECODER=False
    dec_no=0
    SKIP_LAYER=None
    Encoder_2_out=Encoder_2(network,weights_map,Pooling_1.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert Encoder_2_out
    
    
    Pooling_2=network.add_pooling_nd(input=Encoder_2_out.get_output(0),type=trt.PoolingType.MAX, window_size=trt.DimsHW(2, 2))
    Pooling_2.stride_nd=(2,2)
    assert Pooling_2
    
    
    features=128
    layer_name="encoder3.enc3"
    DECODER=False
    dec_no=0
    SKIP_LAYER=None
    Encoder_3_out=Encoder_2(network,weights_map,Pooling_2.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert Encoder_3_out
    
    
    
    Pooling_3=network.add_pooling_nd(input=Encoder_3_out.get_output(0),type=trt.PoolingType.MAX, window_size=trt.DimsHW(2, 2))
    Pooling_3.stride_nd=(2,2)
    assert Pooling_3
    
    
    features=256
    layer_name="encoder4.enc4"
    DECODER=False
    dec_no=0
    SKIP_LAYER=None
    Encoder_4_out=Encoder_2(network,weights_map,Pooling_3.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert Encoder_4_out
    
    
    
    Pooling_4=network.add_pooling_nd(input=Encoder_4_out.get_output(0),type=trt.PoolingType.MAX, window_size=trt.DimsHW(2, 2))
    Pooling_4.stride_nd=(2,2)
    assert Pooling_4
    
    
    features=512
    layer_name="bottleneck.bottleneck"
    DECODER=False
    dec_no=0
    SKIP_LAYER=None
    BOTTLE_out=Encoder_2(network,weights_map,Pooling_4.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert BOTTLE_out
    
    
    features=256
    layer_name="decoder4.dec4"
    DECODER=True
    dec_no=4
    SKIP_LAYER=Encoder_4_out.get_output(0)
    DEC_4_out=Encoder_2(network,weights_map,BOTTLE_out.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert DEC_4_out
    
    
    features=128
    layer_name="decoder3.dec3"
    DECODER=True
    dec_no=3
    SKIP_LAYER=Encoder_3_out.get_output(0)
    DEC_3_out=Encoder_2(network,weights_map,DEC_4_out.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert DEC_3_out
    
    
    features=64
    layer_name="decoder2.dec2"
    DECODER=True
    dec_no=2
    SKIP_LAYER=Encoder_2_out.get_output(0)
    DEC_2_out=Encoder_2(network,weights_map,DEC_3_out.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert DEC_2_out
    
    
    
    features=32
    layer_name="decoder1.dec1"
    DECODER=True
    dec_no=1
    SKIP_LAYER=Encoder_1_out.get_output(0)
    DEC_1_out=Encoder_2(network,weights_map,DEC_2_out.get_output(0),EPS,features,layer_name,DECODER,dec_no,SKIP_LAYER)
    assert DEC_1_out
    
    
    
    Encoder_1_out.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=DEC_1_out.get_output(0))
    
    
def build_FP32_engine(weights_map,batch_size=32):
#     with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser, trt.Runtime(TRT_LOGGER) as runtime:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()#flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
   
    runtime = trt.Runtime(TRT_LOGGER)
#     builder.max_batch_size = batch_size
    config.max_workspace_size = common.GiB(4)
    
#     config.set_flag(trt.BuilderFlag.TF32)

    populate_network(network, weights_map)
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan),builder,config,network




def main(image):
    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    MODEL= student_model.UNet_SAB_STUDENT()
    checkpoint=torch.load(MODEL_PATH)
    MODEL.load_state_dict(checkpoint["state_dict"])
    MODEL.eval()
    weights_map=MODEL.state_dict()

    # Do inference with TensorRT.

    engine,builder,config,network= build_FP32_engine(weights_map)
    
    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()


    image=np.array(image).astype(np.float32)
    image=image/255


    input_image=cv2.resize(image,(512,512),interpolation = cv2.INTER_CUBIC)
    plt.imshow(input_image)

    
    
   
    time_value=[]

    start_time = datetime.datetime.now()
    
    iterations=1000
    for i in range(iterations):

        pagelocked_buffer=inputs[0].host
        np.copyto(pagelocked_buffer, input_image.ravel())

        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        
        
    end_time = datetime.datetime.now()

    time_diff = end_time - start_time

    execution_time = time_diff.total_seconds() * 1000
    execution_time=execution_time/iterations

#     cache=config.get_timing_cache()
#     cache1=cache.serialize()
#     print(cache1)
#     serialized_engine = builder.build_serialized_network(network, config)
#     with open("XPNET-FP32.engine", "wb") as f:
#         f.write(serialized_engine)
    
    
    
    
    return output,execution_time,input_image
if __name__ == '__main__':
    image=Image.open("./EndoCV22_round1_test_seq9_460.jpg")
#     plt.imshow(image)
#     plt.show()
    output_image,execution_time,input_image=main(image)
    print(execution_time,"ms :Float 32,per frame execution_time")
