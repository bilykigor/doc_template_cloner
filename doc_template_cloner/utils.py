import os
from typing import List, Tuple,Union

from difflib import SequenceMatcher
import cv2
import numpy as np

import pytesseract
import torch
import torchvision.ops.boxes as bops
from torch_geometric.data import Data

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


from doc2graph.src.data.graph_builder import GraphBuilder
from doc2graph.src.data.feature_builder import FeatureBuilder
from doc2graph.src.models.utils import GATv2ConvM, GATv2Conv
#from torchmetrics.functional.pairwise import pairwise_cosine_similarity, pairwise_euclidean_distance

os.environ['TOKENIZERS_PARALLELISM']='false'

def find_intersected_boxes(target_box: Tuple[int, int, int, int], 
                           boxes: List[Tuple[int, int, int, int]], 
                           threadhold=0.0) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Finds intersected boxes with target_box. Returns also metric of intersection."""
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
        print(f'Failed to find_intersected_boxes: {exception}')
        print(target_box, boxes)
    
    matched_boxes = sorted(matched_boxes, key=lambda tup: tup[1])
    
    return matched_boxes


def get_relative_boxes_orientation(anchor: Tuple[int, int, int, int],
                                   box: Tuple[int, int, int, int])->List:
    """Categorizes two boxes relative orientation.
    
    Return: 
    relation -- list with 4 options ['down','up','right','left']
    """
    anchor_x = 0.5*(anchor[0]+anchor[2])
    anchor_y = 0.5*(anchor[1]+anchor[3])
    box_x = 0.5*(box[0]+box[2])
    box_y = 0.5*(box[1]+box[3])
    
    relation = []
    if box_y>=anchor[3]:
        relation.append('down')
    if anchor_y>=box[3]:
        relation.append('up')
    if box_x>=anchor[2]:
        relation.append('right')
    if anchor_x>=box[2]:
        relation.append('left')
        
    return relation


def relative_box(anchor_point: Tuple[int, int, int, int], 
                 box: Tuple[int, int, int, int], 
                 is_relative=True)-> Tuple[int, int, int, int]:
    """Calculates relative coordinates of box with respect to anchor_point."""
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
    

def l2_distance(point1: Tuple[int, int],
                point2: Tuple[int, int])->float:
    """L2 distance between points."""
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def merge_points(loc: List[Tuple[int, int]], 
                 l2_threadhold=3)->List[Tuple[int, int]]:
    """Merge close points."""
    
    if len(loc)<2:
        return loc
    
    points=[]
    for l in loc:
        found = False
        for point in points:
            if l2_distance(l,point)<l2_threadhold:
                found=True 
                break
        if not found:
            points.append(l)
            
    return points


def merge_boxes(loc: List[Tuple[int, int, int, int]],
                l2_threadhold=3)->List[Tuple[int, int, int, int]]:
    """Merges close boxes."""
    
    boxes=[]
    if len(loc)<2:
        return loc
    
    for l in loc:
        found = False
        for box in boxes:
            if l2_distance(l[:2],box[:2])<l2_threadhold:
                if l2_distance(l[2:],box[2:])<l2_threadhold:
                    found=True 
                    break
        if not found:
            boxes.append(l)
            
    return boxes


def find_subboxes_by_axis(axis: str, 
                          target_box: Tuple[int, int, int, int], 
                          boxes: List[Tuple[int, int, int, int]], 
                          threadhold=0.9)->List[Tuple[int, int, int, int]]:
    """Finds which of boxes are intersected with target_box by x or y direction."""
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


def find_segment(static_crop: Image, 
                 target_image: Image, 
                 threshold = 0.85,
                 text_threshold = 0.6)->Union[Tuple[int, int, int, int], None]:
    """Finds static_crop image in target_image."""
    
    static_crop_cv2 = cv2.cvtColor(np.array(static_crop), cv2.COLOR_RGB2BGR)
    target_image_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    
    w, h = static_crop_cv2.shape[:-1]
    
    # res = cv2.matchTemplate(target_image_cv2, static_crop_cv2, cv2.TM_CCOEFF_NORMED)
    # loc = np.where(res >= threshold)
    # loc = list(zip(*loc[::-1]))
    # loc = merge_points(loc)
    
    static_crop_cv2 = static_crop_cv2[:, :, :3]
    target_image_cv2 = target_image_cv2[:, :, :3]
    
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
        return None
    
    if len(loc)>1:
        print('Segment found multiple times')
        print(loc)
        plt.imshow(Image.fromarray(static_crop_cv2))
        return None
    
    #print('Segment found')
    pt = loc[0]
    tupleOfTuples = (pt, (pt[0] + h, pt[1] + w))
    found_box = list(sum(tupleOfTuples, ()))
    
    # Compare text in source and target images 
    res = pytesseract.image_to_data(static_crop, lang='eng', config='--psm 7',output_type=pytesseract.Output.DICT)
    static_crop_text = ' '.join(res['text']).strip()
     
    res = pytesseract.image_to_data(target_image.crop(found_box), lang='eng', config='--psm 7',output_type=pytesseract.Output.DICT)
    static_anchor_target_text = ' '.join(res['text']).strip()
    
    s = SequenceMatcher(None, static_crop_text, static_anchor_target_text)
    
    if s.ratio()<text_threshold:
        print(f'Text not match: {static_crop_text} -> {static_anchor_target_text}', s.ratio())
        return None
    
    print('IMAGES', static_crop_text,'--->',static_anchor_target_text)
           
    return found_box


def find_segment_with_graph(static_anchor_source: Tuple[int, int, int, int], 
                            source_word_boxes,
                            target_word_boxes,
                            pairwise_distance,
                            threshold = 0.5,
                            text_threshold = 0.8,
                            is_textbox = False)->Union[Tuple[int, int, int, int], None]:
    
    if not is_textbox:
        static_anchor_source_box = find_intersected_boxes(static_anchor_source, source_word_boxes['boxes'], 0.7)
        
        if len(static_anchor_source_box)>1:
            raise #we dont expect to have multiple boxes intersected with static_anchor_source
        
        if static_anchor_source_box:
            static_anchor_source_box = static_anchor_source_box[0][0]
        else: 
            print('segment not found', static_anchor_source)
            return None
    else:
        static_anchor_source_box = static_anchor_source
    
    ix=-1
    for ix, box in enumerate(source_word_boxes['boxes']):
        if box == static_anchor_source_box:
            break
    
    #print('***********')
    #print('static_anchor_source_box',static_anchor_source_box)
    
    if ix<0:
        return None
        
    d = min(pairwise_distance[ix])
    target_ix = np.argmin(pairwise_distance[ix])  
        
    source_text = source_word_boxes['words'][ix]
    target_text = target_word_boxes['words'][target_ix]
    
    #print(source_text,target_text, ix, target_ix)
    
    s = SequenceMatcher(None, source_text, target_text)

    if s.ratio()<text_threshold:
        print(f'Text not match: {source_text} -> {target_text}', s.ratio())
        return None
    else:
    #if d<threshold:
        static_anchor_target = target_word_boxes['boxes'][target_ix]
        
        if not is_textbox:
            static_anchor_target = relative_box(static_anchor_target,relative_box(static_anchor_source_box,static_anchor_source),is_relative=False)
        
        print('GRAPH', source_word_boxes['words'][ix],'--->',target_word_boxes['words'][target_ix])
        
        # print('static_anchor_source->static_anchor_target', 
        #         d, 
        #         static_anchor_source,
        #         static_anchor_target, 
        #         )
        return static_anchor_target
    #else:
    #    print('segment NOT FOUND',  d, source_word_boxes['words'][ix], source_word_boxes['boxes'][ix])
                     

def get_intersection(box1: Tuple[int, int, int, int],
                     box2: Tuple[int, int, int, int])->Union[Tuple[int, int, int, int], None]:
    """Returns intersection of two rectangles."""
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


def find_bottom_edge(source_image: Image,
                     target_image: Image,
                     variable_many_bbox_source: Tuple[int, int, int, int],
                     statis_bboxes_source: List[Tuple[int, int, int, int]])->Union[int, None]:
    """Finds bottom edge for boxes with variable size."""
    
    down_static_anchor_source = find_subboxes_by_axis('y', variable_many_bbox_source, statis_bboxes_source, threadhold=0.5)
    down_static_anchor_source = [x for x in down_static_anchor_source if x[1]>=variable_many_bbox_source[3]-10] #below boxes
    if down_static_anchor_source:
        down_static_anchor_source = down_static_anchor_source[0]
    
    if down_static_anchor_source:
        static_crop = source_image.crop(down_static_anchor_source)
        down_static_anchor_target = find_segment(static_crop, target_image, threshold = 0.85)
        
        if not down_static_anchor_target:
            return None   
                  
        return down_static_anchor_target[1]
    else:  
        return None       
  
  
def find_bottom_edge_with_graph(
    source_image,
    target_image,
    source_word_boxes,
    target_word_boxes,
    pairwise_distance, 
    variable_many_bbox_source: Tuple[int, int, int, int],
    statis_bboxes_source: List[Tuple[int, int, int, int]])->Union[int, None]:
    """Finds bottom edge for boxes with variable size."""
    
    down_static_anchor_source = find_subboxes_by_axis('y', variable_many_bbox_source, statis_bboxes_source, threadhold=0.5)
    down_static_anchor_source = [x for x in down_static_anchor_source if x[1]>=variable_many_bbox_source[3]-10] #below boxes
    if down_static_anchor_source:
        down_static_anchor_source = down_static_anchor_source[0]
    
    if down_static_anchor_source:
        static_crop = source_image.crop(down_static_anchor_source)
        down_static_anchor_target = find_segment(
            static_crop, 
            target_image, 
            threshold = 0.85)
        if not down_static_anchor_target:
            down_static_anchor_target = find_segment_with_graph(
                            down_static_anchor_source, 
                            source_word_boxes,
                            target_word_boxes,
                            pairwise_distance, 
                            threshold = 0.5)
        
        if not down_static_anchor_target:
            return None   
                  
        return down_static_anchor_target[1]
    else:  
        return None  
   
    
def find_subimages(source_image: Image, 
                 target_image: Image,
                 bboxes_source: List[Tuple[int, int, int, int]],
                 threshold=0.85,
                 text_threshold=0.5)->List[Tuple[int, int, int, int]]:
    """Looks for cropped images from source_image in the target_image, based on bboxes_source coordinates."""
    
    static_boxes_target = []
    
    for static_anchor_source in bboxes_source:
        #Find static_anchor_target
        static_crop = source_image.crop(static_anchor_source)
        static_anchor_target = find_segment(static_crop, target_image, threshold = threshold, text_threshold=text_threshold)
        
        if static_anchor_target is None:
            print('static_anchor_source NOT FOUND', static_anchor_source)
            continue
        
        print('static_anchor_source->static_anchor_target', static_anchor_source, static_anchor_target)
        
        static_boxes_target.append(static_anchor_target) 
        
    print(f'Cloning static {threshold} {len(bboxes_source)} -> {len(static_boxes_target)}')
    
    return static_boxes_target  


def find_similar_boxes(source_image, 
                    target_image,
                    source_word_boxes, 
                    target_word_boxes,
                    bboxes_source: List[Tuple[int, int, int, int]],
                    pairwise_distance, 
                    threshold=0.5)->List[Tuple[int, int, int, int]]:
    """Looks for cropped images from source_image in the target_image, based on bboxes_source coordinates."""
    
    static_boxes_target = []
    for static_anchor_source in bboxes_source:
        static_crop = source_image.crop(static_anchor_source)
        static_anchor_target = find_segment(static_crop, target_image, threshold = 0.85, text_threshold=0.8)
        
        if not static_anchor_target:
            static_anchor_target = find_segment_with_graph(static_anchor_source, 
                            source_word_boxes,
                            target_word_boxes,
                            pairwise_distance,
                            threshold = threshold)
        if static_anchor_target:
            static_boxes_target.append(static_anchor_target) 
            
        print(static_anchor_source,'--->',static_anchor_target)
        
    print(f'Cloning static {threshold} {len(bboxes_source)} -> {len(static_boxes_target)}')
    print(static_boxes_target)
    
    return static_boxes_target  


def clone_relation(source_image: Image, 
                   target_image: Image, 
                   relation_bbox_source: Tuple[int, int, int, int], 
                   statis_bboxes_source: List[Tuple[int, int, int, int]], 
                   static_boxes_target: List[Tuple[int, int, int, int]],
                   variable_one_bboxes_source: List[Tuple[int, int, int, int]],
                   variable_many_bboxes_source: List[Tuple[int, int, int, int]],
                   labeled_boxes_source=None,
                   threshold = 0.85,
                   text_threshold = 0.5):
    
    """Based on labeled boxes in the source image finds respecive "relation" labeled box in the target image.
    
    labeled_boxes_source - dict. {'[x0,y0,x1,y1]':'label',...}"""
    
    static_anchor_source = find_intersected_boxes(relation_bbox_source, statis_bboxes_source,0.1)
    
    #Should be no more then one intersection
    if len(static_anchor_source)>1:
        print('Many static_anchor_source')
        return None
    
    static_anchor_source = static_anchor_source[0][0]

    relation_boxes_target = []
    variable_many_boxes_target = []
    variable_one_boxes_target=[]
    labeled_boxes_target={}
    
    #Find static_anchor_target
    static_crop = source_image.crop(static_anchor_source)
    static_anchor_target = find_segment(static_crop, target_image, threshold = threshold,text_threshold=text_threshold)
    
    #If not found, return empty lists
    if static_anchor_target is None:
        return relation_boxes_target, [], variable_one_boxes_target, variable_many_boxes_target, labeled_boxes_target
    
    #Find relation_bbox_target
    relation_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,relation_bbox_source),is_relative=False) 
    
    #Find relation intersection with variable_many_bboxes
    if variable_many_bboxes_source:
        variable_many_bbox_source = find_intersected_boxes(relation_bbox_source, variable_many_bboxes_source,0.1)
        
        if len(variable_many_bbox_source)>1:
            print('Many variable_many_bbox_source')
            return None
        
        # apply logic to variable many case
        if variable_many_bbox_source:
            variable_many_bbox_source = variable_many_bbox_source[0][0]
            #variable_many_bbox_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
            
            print('variable_many_bbox_source:',variable_many_bbox_source)

            static_in_relation_source = get_intersection(static_anchor_source,relation_bbox_source)
            variable_in_relation_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
            orientation = get_relative_boxes_orientation(static_in_relation_source,variable_in_relation_source)

            if 'down' in orientation:
                variable_many_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_many_bbox_source),is_relative=False)
                bottom_edge = find_bottom_edge(source_image,target_image,variable_many_bbox_source,statis_bboxes_source)
                
                if bottom_edge is None:
                    print('bottom_edge not found')
                    bottom_edge = find_bottom_edge(source_image,target_image,variable_many_bbox_source,variable_one_bboxes_source)
                    
                    if bottom_edge is None:
                        print('bottom_edge not found')
                        bottom_edge = variable_many_bbox_target[3] + 100
                
                print('bottom_edge found', bottom_edge)        
                down_static_anchor_target = find_subboxes_by_axis('y', variable_many_bbox_target, static_boxes_target, threadhold=0.5)
                down_static_anchor_target = [x for x in down_static_anchor_target if variable_many_bbox_target[3]<=x[1]+10]
                if down_static_anchor_target:
                    down_static_anchor_target = down_static_anchor_target[0]
                    
                    if down_static_anchor_target[1]<bottom_edge:
                        bottom_edge = down_static_anchor_target[1]
                        print('bottom_edge corrected', bottom_edge)
                
                #After bottom edge found, need correction
                variable_many_bbox_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
                variable_many_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_many_bbox_source),is_relative=False)
                variable_many_bbox_target[3] = bottom_edge
                relation_bbox_target[3] = bottom_edge              
                variable_many_boxes_target.append(variable_many_bbox_target)
                
                if labeled_boxes_source:
                    labeled_boxes_source_list = [eval(x) for x in labeled_boxes_source.keys()]
                    intersected_labeled_boxes_list = find_intersected_boxes(variable_many_bbox_source, labeled_boxes_source_list,0.7)
                
                    for labeled_box in intersected_labeled_boxes_list:
                        labeled_box = labeled_box[0]
                        
                        labeled_box_target = relative_box(variable_many_bbox_target,relative_box(variable_many_bbox_source,labeled_box),is_relative=False) 
                        labeled_box_target[3] = bottom_edge       
                        labeled_boxes_target[str(labeled_box_target)] = labeled_boxes_source[str(labeled_box)]
            
    relation_boxes_target.append(relation_bbox_target)
    
    #Find relation intersection with variable_one_bboxes_source
    variable_one_bbox_source = find_intersected_boxes(relation_bbox_source, variable_one_bboxes_source,0.1)
    
    if variable_one_bbox_source:
        print('relation_bbox_source->variable_one_bbox_source:',relation_bbox_source, variable_one_bbox_source)
        #print('variable_one_bbox_source:',variable_one_bbox_source)
        for variable_one_bbox in variable_one_bbox_source:
            variable_one_bbox = get_intersection(variable_one_bbox[0],relation_bbox_source)
            variable_one_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_one_bbox),is_relative=False) 
            variable_one_boxes_target.append(variable_one_bbox_target)  
            
            if labeled_boxes_source:
                labeled_boxes_source_list = [eval(x) for x in labeled_boxes_source.keys()]
                intersected_labeled_boxes_list = find_intersected_boxes(variable_one_bbox, labeled_boxes_source_list,0.7)
                
                print('intersected_labeled_boxes_list:', intersected_labeled_boxes_list)
                for labeled_box in intersected_labeled_boxes_list:
                    labeled_box = labeled_box[0]
                    labeled_box_target = relative_box(variable_one_bbox_target,relative_box(variable_one_bbox,labeled_box),is_relative=False) 
                    labeled_boxes_target[str(labeled_box_target)] = labeled_boxes_source[str(labeled_box)]
    
    return relation_boxes_target, [], variable_one_boxes_target, variable_many_boxes_target, labeled_boxes_target
   
   
def clone_labels(source_image, 
                 target_image, 
                 relation_bboxes_source, 
                 statis_bboxes_source, 
                 variable_one_bboxes_source, 
                 variable_many_bboxes_source,
                 labeled_boxes_source=None,
                 threshold=0.85,
                 text_threshold = 0.5):
    relation_boxes = []
    variable_one_boxes = []
    variable_many_boxes = []
    labeled_boxes = []
    
    # Find static boxes
    statis_bboxes_source_intersected = []
    statis_bboxes_source_separate = []
    for statis_bbox in statis_bboxes_source:
        separate = True
        for relation_bbox in relation_bboxes_source:
            intersected = get_intersection(statis_bbox,relation_bbox)
            if intersected:
                separate = False
                statis_bboxes_source_intersected.append(intersected)
        if separate:
            statis_bboxes_source_separate.append(statis_bbox)
                
    static_boxes_target = find_subimages(source_image, 
                                        target_image,
                                        statis_bboxes_source_intersected,
                                        threshold=threshold)

    statis_bboxes_source_intersected.extend(statis_bboxes_source_separate)
    
    #========================================================================
    print('Clonning relation')
    for relation_bbox_source in relation_bboxes_source:
        print('----------------------------------------')
        print(relation_bbox_source)
        relation_boxes_, static_boxes_, variable_one_boxes_, variable_many_boxes_, labeled_boxes_target = clone_relation(
                           source_image, 
                           target_image, 
                           relation_bbox_source, 
                           statis_bboxes_source_intersected, 
                           static_boxes_target,
                           variable_one_bboxes_source, 
                           variable_many_bboxes_source,
                           labeled_boxes_source,
                           threshold)
        print(labeled_boxes_target)
        print('----------------------------------------')
        
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


def clone_relation_with_graphs(
                            source_image, 
                            target_image, 
                            source_word_boxes, 
                            target_word_boxes, 
                            pairwise_distance,
                            relation_bbox_source: Tuple[int, int, int, int], 
                            statis_bboxes_source: List[Tuple[int, int, int, int]], 
                            static_boxes_target: List[Tuple[int, int, int, int]],
                            variable_one_bboxes_source: List[Tuple[int, int, int, int]],
                            variable_many_bboxes_source: List[Tuple[int, int, int, int]],
                            labeled_boxes_source=None,
                            threshold = 0.5):
    
    """Based on labeled boxes in the source image finds respecive "relation" labeled box in the target image.
    
    labeled_boxes_source - dict. {'[x0,y0,x1,y1]':'label',...}"""
    
    static_anchor_source = find_intersected_boxes(relation_bbox_source, statis_bboxes_source,0.1)
    
    #Should be no more then one intersection
    if len(static_anchor_source)>1:
        print('Many static_anchor_source')
        return None
    
    static_anchor_source = static_anchor_source[0][0]

    relation_boxes_target = []
    variable_many_boxes_target = []
    variable_one_boxes_target=[]
    labeled_boxes_target={}
    
    #Find static_anchor_target
    static_crop = source_image.crop(static_anchor_source)
    static_anchor_target = find_segment(static_crop, target_image, threshold = 0.85, text_threshold=0.8)
    if not static_anchor_target:
        static_anchor_target = find_segment_with_graph(static_anchor_source, 
                                                   source_word_boxes, 
                                                   target_word_boxes, 
                                                   pairwise_distance)
    
    #If not found, return empty lists
    if static_anchor_target is None:
        return relation_boxes_target, [], variable_one_boxes_target, variable_many_boxes_target, labeled_boxes_target
    
    #Find relation_bbox_target
    relation_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,relation_bbox_source),is_relative=False) 
    
    #Find relation intersection with variable_many_bboxes
    if variable_many_bboxes_source:
        variable_many_bbox_source = find_intersected_boxes(relation_bbox_source, variable_many_bboxes_source,0.1)
        
        if len(variable_many_bbox_source)>1:
            print('Many variable_many_bbox_source')
            return None
        
        # apply logic to variable many case
        if variable_many_bbox_source:
            variable_many_bbox_source = variable_many_bbox_source[0][0]
            #variable_many_bbox_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
            
            print('variable_many_bbox_source:',variable_many_bbox_source)

            static_in_relation_source = get_intersection(static_anchor_source,relation_bbox_source)
            variable_in_relation_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
            orientation = get_relative_boxes_orientation(static_in_relation_source,variable_in_relation_source)

            if 'down' in orientation:
                variable_many_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_many_bbox_source),is_relative=False)
                bottom_edge = find_bottom_edge_with_graph(
                            source_image, 
                            target_image, 
                            source_word_boxes, 
                            target_word_boxes, 
                            pairwise_distance,
                            variable_many_bbox_source,
                            statis_bboxes_source)
                
                if bottom_edge is None:
                    print('bottom_edge not found')
                    bottom_edge = find_bottom_edge_with_graph(
                            source_image, 
                            target_image, 
                            source_word_boxes, 
                            target_word_boxes, 
                            pairwise_distance,
                            variable_many_bbox_source,
                            variable_one_bboxes_source)
                    
                    if bottom_edge is None:
                        print('bottom_edge not found')
                        bottom_edge = variable_many_bbox_target[3] + 100
                
                print('bottom_edge found', bottom_edge)        
                down_static_anchor_target = find_subboxes_by_axis('y', variable_many_bbox_target, static_boxes_target, threadhold=0.5)
                down_static_anchor_target = [x for x in down_static_anchor_target if variable_many_bbox_target[3]<=x[1]+10]
                if down_static_anchor_target:
                    down_static_anchor_target = down_static_anchor_target[0]
                    
                    if down_static_anchor_target[1]<bottom_edge:
                        bottom_edge = down_static_anchor_target[1]
                        print('bottom_edge corrected', bottom_edge)
                
                #After bottom edge found, need correction
                variable_many_bbox_source = get_intersection(variable_many_bbox_source,relation_bbox_source)
                variable_many_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_many_bbox_source),is_relative=False)
                variable_many_bbox_target[3] = bottom_edge
                relation_bbox_target[3] = bottom_edge              
                variable_many_boxes_target.append(variable_many_bbox_target)
                
                if labeled_boxes_source:
                    labeled_boxes_source_list = [eval(x) for x in labeled_boxes_source.keys()]
                    intersected_labeled_boxes_list = find_intersected_boxes(variable_many_bbox_source, labeled_boxes_source_list,0.7)
                
                    for labeled_box in intersected_labeled_boxes_list:
                        labeled_box = labeled_box[0]
                        
                        labeled_box_target = relative_box(variable_many_bbox_target,relative_box(variable_many_bbox_source,labeled_box),is_relative=False) 
                        labeled_box_target[3] = bottom_edge       
                        labeled_boxes_target[str(labeled_box_target)] = labeled_boxes_source[str(labeled_box)]
            
    relation_boxes_target.append(relation_bbox_target)
    
    #Find relation intersection with variable_one_bboxes_source
    variable_one_bbox_source = find_intersected_boxes(relation_bbox_source, variable_one_bboxes_source,0.1)
    
    if variable_one_bbox_source:
        print('relation_bbox_source->variable_one_bbox_source:',relation_bbox_source, variable_one_bbox_source)
        #print('variable_one_bbox_source:',variable_one_bbox_source)
        for variable_one_bbox in variable_one_bbox_source:
            variable_one_bbox = get_intersection(variable_one_bbox[0],relation_bbox_source)
            variable_one_bbox_target = relative_box(static_anchor_target,relative_box(static_anchor_source,variable_one_bbox),is_relative=False) 
            variable_one_boxes_target.append(variable_one_bbox_target)  
            
            if labeled_boxes_source:
                labeled_boxes_source_list = [eval(x) for x in labeled_boxes_source.keys()]
                intersected_labeled_boxes_list = find_intersected_boxes(variable_one_bbox, labeled_boxes_source_list,0.7)
                
                print('intersected_labeled_boxes_list:', intersected_labeled_boxes_list)
                for labeled_box in intersected_labeled_boxes_list:
                    labeled_box = labeled_box[0]
                    labeled_box_target = relative_box(variable_one_bbox_target,relative_box(variable_one_bbox,labeled_box),is_relative=False) 
                    labeled_boxes_target[str(labeled_box_target)] = labeled_boxes_source[str(labeled_box)]
    
    return relation_boxes_target, [], variable_one_boxes_target, variable_many_boxes_target, labeled_boxes_target
   
  
def clone_labels_with_graphs(source_image, 
                            target_image, 
                            source_file,
                            target_file,
                            source_word_boxes, 
                            target_word_boxes, 
                            relation_bboxes_source, 
                            statis_bboxes_source, 
                            variable_one_bboxes_source, 
                            variable_many_bboxes_source,
                            labeled_boxes_source=None,
                            threshold=0.85,
                            text_threshold = 0.5):
    # Calc graph features
    #========================================================================
    gb = GraphBuilder(data_type = 'word_boxes',
                edge_type = 'fully',
                node_granularity = 'gt',
                word_to_word='word_to_word')
    
    fb = FeatureBuilder(add_embs = True,
                        add_mask = True,
                        add_size=False)
    
    graphs, _, _, features = gb.get_graph(
                [source_file,target_file], 
                'CUSTOM',
                [source_word_boxes, target_word_boxes],
                1)
    
    chunks, _ = fb.add_features(graphs, features) 
    conv = GATv2Conv(sum(chunks), sum(chunks), edge_dim = 2, heads=1, dropout = 0, v2 = True, add_self_loops = False, aggr='add', bias=False)

    graph_features=[]
    for graph in graphs:
        geometric_graph = Data(x=graph.ndata['feat'], 
                            edge_index= torch.stack(graph.edges(), dim=0).to(torch.int64), 
                            edge_attr= graph.edata['feat'])
        x, edge_index, edge_attr = geometric_graph.x, geometric_graph.edge_index, geometric_graph.edge_attr
        x_new = conv(x, edge_index)
        
        graph_features.append(x_new)
        
    #pairwise_distance = pairwise_euclidean_distance(graph_features[0], graph_features[1]).detach().numpy()
    #pairwise_distance[np.isnan(pairwise_distance)] = 0
    def features_dist(a,b):
        n = len(a)//2
        res1 = np.linalg.norm(a[:n]-b[:n])
        res2 = np.linalg.norm(a[n:]-b[n:])
        return res1*0.8+res2*0.2

    pairwise_dist = pairwise_distances(graph_features[0], graph_features[1] ,metric = features_dist)

    #========================================================================
    relation_boxes = []
    variable_one_boxes = []
    variable_many_boxes = []
    labeled_boxes = []
    
    # Find static boxes
    statis_bboxes_source_intersected = []
    statis_bboxes_source_separate = []
    for statis_bbox in statis_bboxes_source:
        separate = True
        for relation_bbox in relation_bboxes_source:
            intersected = get_intersection(statis_bbox,relation_bbox)
            if intersected:
                separate = False
                statis_bboxes_source_intersected.append(intersected)
        if separate:
            statis_bboxes_source_separate.append(statis_bbox)
   
    static_boxes_target = find_similar_boxes(source_image, 
                                        target_image,
                                        source_word_boxes, 
                                        target_word_boxes,
                                        statis_bboxes_source_intersected,
                                        pairwise_dist)

    statis_bboxes_source_intersected.extend(statis_bboxes_source_separate)
    
    #========================================================================
    print('Clonning relation')
    for relation_bbox_source in relation_bboxes_source:
        print('----------------------------------------')
        print(relation_bbox_source)
        relation_boxes_, static_boxes_, variable_one_boxes_, variable_many_boxes_, labeled_boxes_target = clone_relation_with_graphs(
                           source_image, 
                           target_image, 
                           source_word_boxes, 
                           target_word_boxes,
                           pairwise_dist,
                           relation_bbox_source, 
                           statis_bboxes_source_intersected, 
                           static_boxes_target,
                           variable_one_bboxes_source, 
                           variable_many_bboxes_source,
                           labeled_boxes_source,
                           threshold)
        print(labeled_boxes_target)
        print('----------------------------------------')
        
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
