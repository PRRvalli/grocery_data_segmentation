Tp = np.zeros((11,),dtype = 'int')
Fp = np.zeros((11,),dtype = 'int')
Fn = np.zeros((11,),dtype = 'int')
Tn = np.zeros((11,),dtype = 'int')
Total = np.zeros((11,1), dtype = 'int')


pred_list = dict()


for i in test_list:
  name = i
  
  boxes = np.array(dpred_box[name],dtype='int')
  pred = np.array(dclass_box[name],dtype='int')
  score = np.array(dpred_scores[name],dtype='float') # this has the confidence values 
  true_boxes, true_class, true_box_area = true_utils(name)
  pred_boxes, pred_class, pred_box_area = pred_util(pred, boxes,scale = 2)
  
  
  if(len(true_class)==0):
    for n in range(0,len(pred_boxes)):
      class_num = pred_class[n]
      Tn[class_num] +=1 
      
  else:  
    gt_vector = np.ones((len(true_boxes)))
    for n in range(0,len(pred_boxes)):

      iou = compute_iou(np.array(pred_boxes[n],dtype='int'), np.array(true_boxes,dtype='int'), int(pred_box_area[n]), np.array(true_box_area,dtype='int'))
      pos = np.argmax(iou)
      if(iou[pos] > 0.5):
        # true positive
        if(true_class[pos] == pred_class[n]):
          if (gt_vector[pos] ==1):
            #true positive
            class_num = pred_class[n]
            confidence = score[n][class_num]
            Tp[class_num]+=1
            Total[class_num] += 1
            if str(class_num) in pred_list:
              pred_list[str(class_num)].append([confidence,1]) 
            else:
              pred_list[str(class_num)]=[[confidence,1]]
            
          else:
            #False Positive
            class_num = pred_class[n]
            confidence = score[n][class_num]
            Fp[class_num]+=1
            if str(class_num) in pred_list:
              pred_list[str(class_num)].append([confidence,0]) 
            else:
              pred_list[str(class_num)]=[[confidence,0]]

      else:
        # False positive
        if(true_class[pos] == pred_class[n]):
          class_num = pred_class[n]
          confidence = score[n][class_num]
          Fp[class_num]+=1
          if str(class_num) in pred_list:
              pred_list[str(class_num)].append([confidence,0]) 
          else:
              pred_list[str(class_num)]=[[confidence,0]]
          
          
          
    for j in range(0,len(gt_vector)):
      if(gt_vector[j]==1):
        class_num = true_class[j]
        Fn[class_num] +=1 
        Total[class_num] += 1
        
        
# this compute the area under the precision recall curve
# pred_list is a dictinary that consist of the confidence score and the decision of all boxes corresponding to each class.

total = 0
for i in range(1,11):
  class_num = i
  x = pred_list[str(class_num)]
  x.sort(key=first_element, reverse = True)
  tp_cumsum = np.cumsum(np.array(x)[:,1])
  pos = np.cumsum(np.ones((len(x),),dtype='int'))
  

  precision = tp_cumsum/pos
  recall = tp_cumsum/(Tp[class_num]+Fn[class_num])
  
  '''plt.plot(recall,precision)
  plt.show()'''
  # max_recall = np.max(recall)
  Ap = 0
  for i in np.linspace(0,1,11):
    if np.sum(recall>=i) == 0:
      p = 0
    else:
      p = np.max(precision[recall>=i])
    Ap += p/11
  # print(Ap)
  total += Ap/10
  
# mean average precision
print(total)
  
   

  


