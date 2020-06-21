import numpy as np
import tensorflow as tf
from matcher import HungarianMatcher

class DETRLosses():
  def __init__(self,num_classes = 2):
    super(DETRLosses, self).__init__()     
    self.weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 2}     
    self.matcher = HungarianMatcher()
     
    self.num_classes = num_classes+1
    self.empty_weight = np.ones([self.num_classes])
    self.eos_coef = 0.1
    self.empty_weight[-1] = self.eos_coef
     
    self.empty_weight = tf.convert_to_tensor(self.empty_weight) 
    

  def _get_src_permutation_idx(self, indices):
    # permute predictions following indices
     
    batch_idx = tf.concat([tf.fill(src.shape, i) for i, (src, _) in enumerate(indices)],axis=0)
    batch_idx = tf.cast(batch_idx,tf.int64)
     
    src_idx = tf.concat([src for (src, _) in indices],axis=0)
     
    return batch_idx, src_idx

  def _get_tgt_permutation_idx(self, indices):
    # permute targets following indices
    batch_idx =tf.concat([tf.fill(tgt.shape, i) for i, (_, tgt) in enumerate(indices)],axis=0)
    tgt_idx = tf.concat([tgt for (_, tgt) in indices],axis=0)
    return batch_idx, tgt_idx

  def bbox_loss(self,outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
     
    idx = self._get_src_permutation_idx(indices)   
    idx_t = tf.transpose(tf.stack((idx))) 
    src_boxes = tf.gather_nd(outputs['pred_boxes'],idx_t)    
    
    target_boxes = tf.concat([tf.gather(t['bboxes'],i) for t, (_, i) in zip(targets, indices)], axis=0)
    target_boxes = tf.cast(target_boxes,tf.float32)
   
    loss_bbox = tf.math.abs(tf.math.subtract(src_boxes,target_boxes)) #L1     
    
    losses = {}
    losses['loss_bbox'] = tf.math.reduce_sum(loss_bbox) / num_boxes

    loss_giou = tf.linalg.diag_part(1-(self.matcher.generalized_box_iou(
        self.matcher.box_cxcywh_to_xyxy(src_boxes),
        self.matcher.box_cxcywh_to_xyxy(target_boxes))))
 
    losses['loss_giou'] = tf.math.reduce_sum(loss_giou) / num_boxes
    print("LOSS_BBOX",losses['loss_bbox'])
    print("LOSS_GIOU",losses['loss_giou'])
    return losses 

  def class_loss(self,outputs, targets, indices, num_boxes):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']

    idx = self._get_src_permutation_idx(indices)
    idx_t = tf.transpose(tf.stack((idx)))
    # print("IDX_t",idx_t)
    target_classes_o = tf.concat([tf.gather(t['labels'],J) for t, (_, J) in zip(targets, indices)],axis=0) 
    target_classes_o = tf.cast(target_classes_o,tf.int32)
    
     
    target_classes = tf.fill(src_logits.shape[:2], self.num_classes-1)   

     
           
        
    target_classes =  tf.tensor_scatter_nd_update(target_classes,idx_t,target_classes_o)
    # print("targ classes",target_classes.shape)
    # print("src_logits",src_logits.shape)
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    
    loss_ce = cce(target_classes,src_logits).numpy()
    print("LOSS_CE",loss_ce) 
    losses = {'loss_ce': loss_ce}
    return losses       

  def combined_loss_fn(self,outputs,targets):
    # Class Loss and BBOX loss
    indices = self.matcher.runMatch(outputs, targets)
    print("indices",indices)
    #print("indices",indices)
    # Compute the average number of target boxes accross all nodes, for normalization purposes
    num_boxes = sum(len(t["labels"]) for t in targets)
    # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    num_boxes = tf.convert_to_tensor([num_boxes],dtype=tf.float64) 
    # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    num_boxes = tf.clip_by_value(num_boxes,1,tf.float32.max).numpy()
    # print("num_boxes")
    bboxLoss = self.bbox_loss(outputs,targets,indices, num_boxes)
    classLoss = self.class_loss(outputs,targets,indices, num_boxes)

    combined_loss = {**bboxLoss, **classLoss}
    # combined_loss = classLoss
    loss = sum(combined_loss[k] * self.weight_dict[k] for k in combined_loss.keys() if k in self.weight_dict)
    print("LOSS",combined_loss,loss)
    return loss
 

