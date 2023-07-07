import numpy as np
import cv2

import torch
import torchvision.ops.boxes as bops

from PIL import ImageDraw,ImageFont,Image
from matplotlib import pyplot as plt


def find_intersected(target_box, boxes, threadhold=0.0):
    """looks for intersected boxes with target_box"""
    matched_boxes=[]
    
    try:
        for box in boxes:
            area = bops.box_iou(torch.tensor([box], dtype=torch.float), torch.tensor([target_box], dtype=torch.float))
            area_a = bops.box_area(torch.tensor([box], dtype=torch.float))
            area_b = bops.box_area(torch.tensor([target_box], dtype=torch.float))
            res = area/(1+area)*(area_a+area_b)
            area = res/min([area_a,area_b])
            if area>threadhold:
                matched_boxes.append((box,area))
    except Exception as exception:
        print(f'Error find_intersected: {exception}')
        print(target_box, boxes)
    
    matched_boxes = sorted(matched_boxes, key=lambda tup: tup[1])
    
    #if len(matched_boxes)>0:
    #    return matched_boxes[-1][0]
                
    return matched_boxes


def get_relative_orientation(anchor,box):
    """sumary_line
    
    Arguments:
    anchor -- box with 4 coordinates (x0,y0,x1,y1)
    box    -- box with 4 coordinates (x0,y0,x1,y1)
    
    Return: 
    relation -- list with 4 options ['down','up','right','left']
    """
    
    relation = []
    
    anchor_x = 0.5*(anchor[0]+anchor[2])
    anchor_y = 0.5*(anchor[1]+anchor[3])
    box_x = 0.5*(box[0]+box[2])
    box_y = 0.5*(box[1]+box[3])
    
    if box_y>=anchor[3]:
        relation.append('down')
    if anchor_y>=box[3]:
        relation.append('up')
    if box_x>=anchor[2]:
        relation.append('right')
    if anchor_x>=box[2]:
        relation.append('left')
        
    return relation


def relative_box(anchor_point, box, is_relative=True):
    # Calculates relative coordinates of box with respect to anchor_point.
    if is_relative:
        return [
            box[0]-anchor_point[0],
            box[1]-anchor_point[1],
            box[2]-anchor_point[0],
            box[3]-anchor_point[1]
                ]
    
    return [
        box[0]+anchor_point[0],
        box[1]+anchor_point[1],
        box[2]+anchor_point[0],
        box[3]+anchor_point[1]
        ]
    

def l2_distance(point1,point2):
    """l2 distance between points"""
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def merge_points(loc):
    """Merge close points
    
    Arguments:
    loc -- list of points (x,y)
    
    Return: 
    points -- list 
    """
    
    points=[]
    if len(loc)<2:
        return loc
    
    for l in loc:
        found = False
        for point in points:
            if l2_distance(l,point)<3:
                found=True 
                break
        if not found:
            points.append(l)
            
    return points


def merge_boxes(loc):
    """Merge close boxes
    
    Arguments:
    loc -- list of boxes [x0,y0,x1,y1]
    
    Return: 
    points -- list 
    """
    
    boxes=[]
    if len(loc)<2:
        return loc
    
    for l in loc:
        found = False
        for box in boxes:
            if l2_distance(l[:2],box[:2])<3:
                if l2_distance(l[2:],box[2:])<3:
                    found=True 
                    break
        if not found:
            boxes.append(l)
            
    return boxes


def find_subboxes_by_axis(axis, target_box, boxes, threadhold=0.9):
    """look which of boxes are inside target_box"""
    matched_boxes=[]
    target_box = target_box.copy() #immutable
    for box in boxes:
        #making same x coordinates
        if axis=='x':
            target_box[0]=box[0]
            target_box[2]=box[2]
        else:
            target_box[1]=box[1]
            target_box[3]=box[3]
        
        area = bops.box_iou(torch.tensor([box], dtype=torch.float), torch.tensor([target_box], dtype=torch.float))
        area_A = bops.box_area(torch.tensor([box], dtype=torch.float))
        area_B = bops.box_area(torch.tensor([target_box], dtype=torch.float))
        
        res = area/(1+area)*(area_A+area_B)
        area = res/min([area_A,area_B])
        
        if area>threadhold:
            matched_boxes.append(box)
            
    if axis=='x':
        matched_boxes = sorted(matched_boxes, key=lambda x: x[0]+x[2])
    if axis=='y':
        matched_boxes = sorted(matched_boxes, key=lambda x: x[1]+x[3])
    
    return matched_boxes

# if type(bbox)==dict:
#         bbox = labelbox_box_to_image_box(bbox, width_scale, heights_scale)
    
# static_crop = source_image.crop(bbox)
# static_crop_cv2 = cv2.cvtColor(np.array(static_crop), cv2.COLOR_RGB2BGR)


def find_segment(static_crop_cv2, target_image_cv2, threshold = 0.85):
    """finds static_crop_cv2 image in target_image_cv2"""
    w, h = static_crop_cv2.shape[:-1]
    
    # res = cv2.matchTemplate(target_image_cv2, static_crop_cv2, cv2.TM_CCOEFF_NORMED)
    # loc = np.where(res >= threshold)
    # loc = list(zip(*loc[::-1]))
    # loc = merge_points(loc)
    
    res = cv2.matchTemplate(target_image_cv2, static_crop_cv2, cv2.TM_CCORR_NORMED)
    _,score,_,point = cv2.minMaxLoc(res)
    
    found_box = None
    
    if score<threshold:
        print('Segment not found')
        plt.imshow(Image.fromarray(static_crop_cv2))
        return None
    
    loc = [point]
    
    if len(loc)==0:
        print('Segment not found')
        plt.imshow(Image.fromarray(static_crop_cv2))
        #cv2.imshow(static_crop_cv2)
        return None
    elif len(loc)>1:
        print('Segment found multiple times')
        print(loc)
        plt.imshow(Image.fromarray(static_crop_cv2))
        return None
    else:    
        #print('Segment found')
        pt = loc[0]
        tupleOfTuples = (pt, (pt[0] + h, pt[1] + w))
        found_box = list(sum(tupleOfTuples, ()))
        
    return found_box


def get_intersection(box1,box2):
    "returns intersection of two rectangles"
    if box1[3]<=box2[1]:
        return None
    
    if box2[3]<=box1[1]:
        return None
    
    if box1[2]<=box2[0]:
        return None
    
    if box2[2]<=box1[0]:
        return None
    
    result = [max(box1[0],box2[0]),max(box1[1],box2[1]),min(box1[2],box2[2]),min(box1[3],box2[3])]
    
    return result


def clone_labels(source_image, 
                 target_image, 
                 relation_bboxes_source, 
                 statis_bboxes_source, 
                 variable_one_bboxes_source, 
                 variable_many_bboxes_source,
                 labeled_boxes_source=None,
                 threshold=0.85):
    target_image_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    
    relation_boxes = []
    variable_one_boxes = []
    variable_many_boxes = []
    labeled_boxes = []
    
    static_boxes_target = clone_static(source_image, 
                                target_image_cv2, 
                                statis_bboxes_source,
                                threshold=threshold)

    print('Clonning relation')
    for relation_bbox_source in relation_bboxes_source:
        relation_boxes_, static_boxes_, variable_one_boxes_, variable_many_boxes_, labeled_boxes_target = clone_relation(relation_bbox_source, 
                           source_image, 
                           target_image_cv2, 
                           statis_bboxes_source, 
                           static_boxes_target,
                           variable_one_bboxes_source, 
                           variable_many_bboxes_source,
                           labeled_boxes_source,
                           threshold)
        print(relation_bbox_source,'-->',labeled_boxes_target)
        
        relation_boxes.extend(relation_boxes_)
        #static_boxes_target.extend(static_boxes_)
        variable_many_boxes.extend(variable_many_boxes_)
        variable_one_boxes.extend(variable_one_boxes_)
        labeled_boxes.append(labeled_boxes_target)
        
    relation_boxes = merge_boxes(relation_boxes)
    #static_boxes_target = merge_boxes(static_boxes_target)
    variable_one_boxes = merge_boxes(variable_one_boxes)
    variable_many_boxes = merge_boxes(variable_many_boxes)
    #labeled_boxes = merge_boxes(labeled_boxes)
    
    return relation_boxes, static_boxes_target, variable_one_boxes, variable_many_boxes, labeled_boxes

        
def find_bottom_edge(source_image,target_image_cv2,variable_many_bbox_source,statis_bboxes_ll):
    down_static_anchor_source = find_subboxes_by_axis('y', variable_many_bbox_source, statis_bboxes_ll, threadhold=0.5)
    down_static_anchor_source = [x for x in down_static_anchor_source if variable_many_bbox_source[3]<=x[1]+10]
    if down_static_anchor_source:
        down_static_anchor_source = down_static_anchor_source[0]
    
    if down_static_anchor_source:
        static_crop = source_image.crop(down_static_anchor_source)
        static_crop_cv2 = cv2.cvtColor(np.array(static_crop), cv2.COLOR_RGB2BGR)
        down_static_anchor_target = find_segment(static_crop_cv2, target_image_cv2, threshold = 0.85)
        
        if not down_static_anchor_target:
            return None   
                  
        return down_static_anchor_target[1]
    else:  
        return None       
    
    
def clone_static(source_image, 
                target_image_cv2, 
                statis_bboxes_ll,
                threshold=0.85):
    
    static_boxes = []
    
    #Find relation intersection with statis_bboxes
    for static_anchor_source in statis_bboxes_ll:
        #Find static_anchor_target
        static_crop = source_image.crop(static_anchor_source)
        static_crop_cv2 = cv2.cvtColor(np.array(static_crop), cv2.COLOR_RGB2BGR)
        static_anchor_target = find_segment(static_crop_cv2, target_image_cv2, threshold = threshold)
    
        if static_anchor_target is None:
            continue
    
        static_boxes.append(static_anchor_target) 
    
    return static_boxes


def clone_relation(relation_bbox_source, 
                   source_image, 
                   target_image_cv2, 
                   statis_bboxes_source, 
                   static_boxes_target,
                   variable_one_bboxes_source,
                   variable_many_bboxes_source,
                   labeled_boxes_source=None,
                   threshold=0.85):
    
    """labeled_boxes_source - dict. {'[x0,y0,x1,y1]':'label',...}"""
    
    relation_boxes_target = []
    #static_boxes_target = []
    variable_many_boxes_target = []
    variable_one_boxes_target=[]
    labeled_boxes_target={}
    
    #Find relation intersection with statis_bboxes
    static_anchor_source = find_intersected(relation_bbox_source, statis_bboxes_source,0.1)
    
    #Shold be no more then one intersection
    if len(static_anchor_source)>1:
        print('Many static_anchor_source')
        return None
    
    #Find static_anchor_target
    static_anchor_source = static_anchor_source[0][0]
    static_crop = source_image.crop(static_anchor_source)
    static_crop_cv2 = cv2.cvtColor(np.array(static_crop), cv2.COLOR_RGB2BGR)
    static_anchor_target = find_segment(static_crop_cv2, target_image_cv2, threshold = threshold)
    
    #If not found, return empty lists
    if static_anchor_target is None:
        return relation_boxes_target, [], variable_one_boxes_target, variable_many_boxes_target, labeled_boxes_target
    #static_boxes_target.append(static_anchor_target)
    
    #Find relation_bbox_target
    relation_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,relation_bbox_source),is_relative=False) 
    
    #Find relation intersection with variable_many_bboxes
    if variable_many_bboxes_source:
        variable_many_bbox_source = find_intersected(relation_bbox_source, variable_many_bboxes_source,0.1)
        
        if len(variable_many_bbox_source)>1:
            print('Many variable_many_bbox_source')
            return None
        
        # apply logic to variable many case
        if variable_many_bbox_source:
            variable_many_bbox_source = variable_many_bbox_source[0][0]
            
            static_in_relation_source = get_intersection(static_anchor_source,relation_bbox_source)
            variable_in_relation_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
            orientation = get_relative_orientation(static_in_relation_source,variable_in_relation_source)

            if 'down' in orientation:
                variable_many_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_many_bbox_source),is_relative=False)
                
                bottom_edge = find_bottom_edge(source_image,target_image_cv2,variable_many_bbox_source,statis_bboxes_source)
                
                if bottom_edge is None:
                    bottom_edge = find_bottom_edge(source_image,target_image_cv2,variable_many_bbox_source,variable_one_bboxes_source)
                    
                    if bottom_edge is None:
                        bottom_edge = variable_many_bbox_target[3] + 100
                        
                down_static_anchor_target = find_subboxes_by_axis('y', variable_many_bbox_target, static_boxes_target, threadhold=0.5)
                down_static_anchor_target = [x for x in down_static_anchor_target if variable_many_bbox_target[3]<=x[1]+10]
                if down_static_anchor_target:
                    down_static_anchor_target = down_static_anchor_target[0]
                    
                    bottom_edge = min(bottom_edge,down_static_anchor_target[1])
                
                
                variable_many_bbox_target[3] = bottom_edge
                relation_bbox_target[3] = bottom_edge              
                variable_many_boxes_target.append(variable_many_bbox_target)
                
                #looking for labeled boxes
                if labeled_boxes_source:
                    labeled_boxes_source_list = [eval(x) for x in labeled_boxes_source.keys()]
                    intersected_labeled_boxes_list = find_intersected(variable_in_relation_source, labeled_boxes_source_list,0.1)
                
                    for labeled_box in intersected_labeled_boxes_list:
                        labeled_box = labeled_box[0]
                        
                        labeled_box_target = relative_box(variable_many_bbox_target,relative_box(variable_many_bbox_source,labeled_box),is_relative=False) 
                        labeled_box_target[3] = bottom_edge       
                        labeled_boxes_target[str(labeled_box_target)] = labeled_boxes_source[str(labeled_box)]
            
    relation_boxes_target.append(relation_bbox_target)
    
    #Find relation intersection with variable_one_bboxes_source
    variable_one_bbox_source = find_intersected(relation_bbox_source, variable_one_bboxes_source,0.1)
    
    print('variable_one_bbox_source:',variable_one_bbox_source)
    for variable_one_bbox in variable_one_bbox_source:
        variable_one_bbox = variable_one_bbox[0]
        variable_one_bbox_in_relation = get_intersection(variable_one_bbox,relation_bbox_source)
        variable_one_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_one_bbox),is_relative=False) 
        variable_one_boxes_target.append(variable_one_bbox_target)  
        
        if labeled_boxes_source:
            labeled_boxes_source_list = [eval(x) for x in labeled_boxes_source.keys()]
            intersected_labeled_boxes_list = find_intersected(variable_one_bbox_in_relation, labeled_boxes_source_list,0.1)
            
            print('intersected_labeled_boxes_list:', intersected_labeled_boxes_list)
            for labeled_box in intersected_labeled_boxes_list:
                labeled_box = labeled_box[0]
                labeled_box_target = relative_box(variable_one_bbox_target,relative_box(variable_one_bbox,labeled_box),is_relative=False) 
                labeled_boxes_target[str(labeled_box_target)] = labeled_boxes_source[str(labeled_box)]
    
    return relation_boxes_target, [], variable_one_boxes_target, variable_many_boxes_target, labeled_boxes_target
