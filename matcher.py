from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np 
import tensorflow as tf


class HungarianMatcher():
  def __init__(self,cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):    
    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou
    assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

  def runMatch(self,outputs,targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        output_pred_logits = outputs['pred_logits'] #(1, 100, 3) 
        output_bbox = outputs['pred_boxes'] # (1, 100, 4)
        bs,num_queries = output_pred_logits.shape[:2]        
        

        # We flatten to compute the cost matrices in a batch         
        out_prob = tf.reshape(output_pred_logits,[bs*num_queries,-1])
        
        out_prob = tf.keras.layers.Softmax()(out_prob)    
        out_bbox = tf.reshape(output_bbox,[output_bbox.shape[0]*output_bbox.shape[1],-1])        
        # Also concat the target labels and boxes        
        tgt_ids  = tf.concat([v["labels"] for v in targets], axis=0)          
        tgt_bbox = tf.concat([v["bboxes"] for v in targets], axis=0)
        # print("tgt_ids",tgt_ids)
        # print("tgt_bbox",tgt_bbox)
        # for i in tgt_bbox:
        #   ls =  rescale_bboxes(i,(640,512))
        #   ls2 =  self.box_xyxy_to_cxcywh(ls)   
        #   # ls3 = [ [i[0]-(i[2])/2,i[1]-i[3]/2,i[2],i[3]] for i in ls2]      
        #   print(ls2)
        # kek
        # print("len_tgtbox",len(tgt_bbox))         
        tgt_bbox = tf.cast(tgt_bbox, dtype=tf.float32, name=None)
        # print("tgt",tgt_bbox.shape)
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
         
        # cost_class = -out_prob[:, tgt_ids]
         
        tgt_ids = tf.cast(tgt_ids,tf.int32)
         
        cost_class = tf.gather(-out_prob, tgt_ids,axis=1)        
        # Compute the L1 cost between boxes        
         
       
        # print("len tgt bbox",len(tgt_bbox))
        cost_bbox = cdist(out_bbox.numpy(), tgt_bbox.numpy(), 'euclidean',p=1)    
        
         
        # Compute the giou cost betwen boxes
        cost_giou = -self.generalized_box_iou(self.box_cxcywh_to_xyxy(out_bbox), self.box_cxcywh_to_xyxy(tgt_bbox))
        # print("cost_bbox",cost_bbox.shape)
        # print("cost_class",cost_class.shape)
        # print("cost_giou",cost_giou.shape)
        # Final cost matrix
        C = self.cost_bbox * self.cost_bbox + self.cost_class * self.cost_class + self.cost_giou * cost_giou
       
        C = tf.reshape(C,[bs, num_queries, -1]) 

        
         
        sizes = [len(v["bboxes"]) for v in targets]
        #print(tf.split(C,sizes,axis=-1))
        # for i, c in enumerate(tf.split(C,sizes,axis=-1)):
        #   print(c[i])
         
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(tf.split(C,sizes,axis=-1))]
        # print("indices",indices)
        return [(tf.convert_to_tensor(i,dtype=tf.int64), tf.convert_to_tensor(j,dtype=tf.int64)) for i, j in indices]       


  def box_cxcywh_to_xyxy(self,x):
      x_c, y_c, w, h = tf.unstack(x,axis=-1)
      b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
      # print(tf.stack(b, axis=-1))
      return tf.stack(b, axis=-1)

 
  def box_xyxy_to_cxcywh(self,x):
      x0, y0, x1, y1 = tf.unstack(x,axis=-1)
      b = [(x0 + x1) / 2, (y0 + y1) / 2,
          (x1 - x0), (y1 - y0)]
      return tf.stack(b, axis=-1)

  def box_area(self,boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

  def box_iou(self,boxes1, boxes2):
      area1 = self.box_area(boxes1)
      area2 = self.box_area(boxes2)

      lt = tf.math.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
      #print("lt",lt.shape)
      # kek
      rb = tf.math.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

      wh = tf.clip_by_value(rb - lt,0,tf.float32.max)  # [N,M,2]
      inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

      union = area1[:, None] + area2 - inter

      iou = inter / union
      return iou, union      

  def generalized_box_iou(self,boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = self.box_iou(boxes1, boxes2)

    lt = tf.math.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = tf.math.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    wh = tf.clip_by_value(rb - lt, 0,tf.float32.max) # [N,M,2]
     
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

 

# matcher = HungarianMatcher()  
# matcher.runMatch(out,toyGT)