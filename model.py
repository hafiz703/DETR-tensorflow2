import tensorflow as tf
from transformer import Transformer
import numpy as np
import math

class DETR(tf.keras.Model):
  def __init__(self,num_classes=2,hidden_dim=256, num_encoder_decoder_layers = 6, num_queries=100,num_heads=8,dff=2048):
    super(DETR, self).__init__()
    # Hyperparams
    self.num_classes = num_classes+1

    # Transformer
    self.hidden_dim = hidden_dim 
    self.num_queries = num_queries
    self.transformer_bs = 200
    self.num_heads = num_heads
    self.num_encoder_decoder_layers = num_encoder_decoder_layers
    self.dff = dff
    

    self.backbone = tf.keras.applications.ResNet50(
    include_top=False, input_shape=(None,None,3), weights=None, classes=self.num_classes)
    
    #Layers
    self.linear_class = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", use_bias=True),         
        tf.keras.layers.Dense(self.num_classes)             
    ])
 
    self.query_pos = tf.Variable(tf.random.normal([self.num_queries,1,self.hidden_dim]),dtype=tf.float32) 
    self.row_embed =  tf.Variable(tf.random.normal([self.num_queries//2,self.hidden_dim // 2]),dtype=tf.float32) 
    self.col_embed =  tf.Variable(tf.random.normal([self.num_queries//2,self.hidden_dim // 2]),dtype=tf.float32)
    
    self.transformer = Transformer(pe_input=self.hidden_dim, pe_target=self.num_queries, num_layers=self.num_encoder_decoder_layers, d_model=self.hidden_dim, num_heads=self.num_heads, dff=self.dff,rate=0.1, input_vocab_size=self.hidden_dim, 
                target_vocab_size=self.num_queries)   
    
    #MLP for bbox
    self.linear_bbox = tf.keras.Sequential([
        tf.keras.layers.Dense(self.hidden_dim, activation="relu", use_bias=True),         
        tf.keras.layers.Dense(4)             
    ])

  def call(self, inputs,mask=None): 
    batch_size = inputs.shape[0]

    backbone = self.backbone(inputs)
  
    conv1x1 = tf.keras.layers.Conv2D(self.hidden_dim, (1,1), activation ='relu')(backbone)
    conv1x1 = tf.transpose(conv1x1,perm=[0,3,1,2])
    
    _,dim,H, W = conv1x1.shape 
 
    
    colEmbed = tf.expand_dims(self.col_embed[:W],0) 
    colEmbed = tf.tile(colEmbed,(H,1,1))
   
    rowEmbed = tf.expand_dims(self.row_embed[:H],1) 
    rowEmbed = tf.tile(rowEmbed,(1,W,1))
 
    inp = tf.reshape(conv1x1,[-1,batch_size,self.hidden_dim])

    ######    
    inpShape = tf.shape(inp).numpy()
    divisibleBy = self.num_heads * batch_size * self.num_queries *(self.hidden_dim // self.num_heads)
    if(np.prod(inpShape)%divisibleBy!=0):
      newdim = int(math.ceil(np.prod(inpShape)/divisibleBy) * divisibleBy/(np.prod(inpShape[1:])) - inpShape[0])
      pad = tf.zeros([newdim,batch_size,self.hidden_dim])
      inp = tf.concat([inp,pad],axis=0)       
    ######


    targ = tf.tile(self.query_pos,(1,batch_size,1))
    # targ = tf.expand_dims(targ,1)
    print("targ",targ.shape,inp.shape) # (batch_size, 100, 256)
 
    
    self.out, _ = self.transformer(inp,targ,True,None,None,None) 
 
    class_output  = self.linear_class(self.out)
    bbox_output = self.linear_bbox(self.out)
    bbox_output = tf.keras.activations.sigmoid(bbox_output)
    # print(class_output.shape,bbox_output.shape)
    return {'pred_logits': class_output, 'pred_boxes': bbox_output}

 
 