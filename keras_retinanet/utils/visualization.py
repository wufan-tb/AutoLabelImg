"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .colors import label_color


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
    
def common_draw_img(draw_info, img, labels, scores, boxes, save_path, **kwds):

    plt.rcParams['figure.figsize'] = (19.2, 10.8)
    plt.rcParams['savefig.dpi'] = 100
    linewidth=3
    fontsize=15
    plt.imshow(img)

    for i in range(labels.shape[0]):
        label=int(labels[i])
        score=scores[i]
        box=boxes[i]
        condition_1 = label in draw_info['label_name'].keys()
        try:
            condition_2 = kwds['exist'][i]==True
        except:
            condition_2=True
        
        if  condition_1 and condition_2 :
            ymin = min(img.shape[0]-5,max(5,int(box[draw_info['box_type'].index('ymin')])))
            xmin = min(img.shape[1]-5,max(5,int(box[draw_info['box_type'].index('xmin')])))
            ymax = max(5,min(img.shape[0]-5,int(box[draw_info['box_type'].index('ymax')])))
            xmax = max(5,min(img.shape[1]-5,int(box[draw_info['box_type'].index('xmax')])))
            class_name = str(draw_info['label_name'][label])
            class_color=draw_info['label_color'][label]
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=class_color,
                             linewidth=linewidth)
            plt.gca().add_patch(rect)
            info_txt='{:s} | {:.2f} | '.format(class_name, score)
            info_length=2
            for key in kwds:
                if info_length%3==0:
                    info_txt+='\n'
                info_txt=info_txt+'{}:{:s} | '.format(key,str(kwds[key][i]))
                info_length+=1
            plt.gca().text(min(img.shape[1]-15,max(xmin+1,0)), min(img.shape[0]-10,max(ymin-8,0)),
                           info_txt,
                           bbox=dict(facecolor=class_color, alpha=0.5),
                           fontsize=fontsize, color='white')
        elif label!=-1:
            print('label {} not in info or exist!'.format(label))
            pass
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close()
    
def remove_certain_label(labels,scores,boxes,label_list):
    label_list=np.array(label_list)
    temp=[]
    temp.append(labels)
    temp.append(scores)
    temp.append(boxes)
    for item in label_list:
        the_label=int(item)
        temp[2] = temp[2][temp[0]!=the_label]
        temp[1] = temp[1][temp[0]!=the_label]
        temp[0] = temp[0][temp[0]!=the_label]
    return temp[0],temp[1],temp[2]

def transform_box(boxes,detect_window=None):
    if detect_window==None:
        pass
    else:
        for i in range(len(boxes)):
            boxes[i][0]=boxes[i][0]+detect_window[0]
            boxes[i][1]=boxes[i][1]+detect_window[1]
            boxes[i][2]=boxes[i][2]+detect_window[0]
            boxes[i][3]=boxes[i][3]+detect_window[1]
    return boxes