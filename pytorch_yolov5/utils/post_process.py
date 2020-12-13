#================== def some functions ==================
import cv2,time,os,json,datetime,csv,matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_box_info(draw_info, img, labels, scores, boxes, **kwds):
    for i in range(len(boxes)):
        label=int(labels[i])
        score=round(scores[i],2)
        box=boxes[i]
        mean_score=round(kwds['Mscores'][i],2) if 'Mscores' in kwds.keys() else score
        obj_id='ID:'+str(int(kwds['ID'][i])) if 'ID' in kwds.keys() else ''
        
        if min(score,mean_score)>=draw_info['draw_threshold'][label]:
            ymin = min(img.shape[0]-5,max(5,int(box[draw_info['box_type'].index('ymin')])))
            xmin = min(img.shape[1]-5,max(5,int(box[draw_info['box_type'].index('xmin')])))
            ymax = max(5,min(img.shape[0]-5,int(box[draw_info['box_type'].index('ymax')])))
            xmax = max(5,min(img.shape[1]-5,int(box[draw_info['box_type'].index('xmax')])))
            class_name = str(draw_info['label_name'][label])
            class_color=draw_info['label_color'][label]
            info_txt='{:s}|{}'.format(class_name,obj_id)
            t_size=cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_TRIPLEX, 0.8 , 2)[0]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), class_color, 2)
            cv2.rectangle(img, (xmin, ymin), (xmin + t_size[0]+4, ymin + t_size[1]+10), class_color, -1)
            cv2.putText(img, info_txt, (xmin+2, ymin+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, 0.8, [255,255,255], 2)
    return img
    
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
        try:
            condition = draw_info['if_draw_box'][i]
        except:
            try:
                condition= (kwds['exist'][i]==True)
            except:
                condition = False
        
        if  score>=0 :
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
            info_txt='{:s} | {:.3f} | '.format(class_name, score)
            info_length=2
            for key in kwds:
                if key != 'Counter':
                    if info_length%3==0:
                        info_txt+='\n'
                    info_txt=info_txt+'{:s}: {:s} | '.format(key, str(kwds[key][i]))
                    info_length+=1
            plt.gca().text(min(img.shape[1]-15,max(xmin+1,0)), min(img.shape[0]-10,max(ymin-8,0)),
                               info_txt,
                               bbox=dict(facecolor=class_color, alpha=0.5),
                               fontsize=fontsize, color='white')
        else:
            pass
        
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close()
    
def box_pre_select(labels,scores,boxes,label_list,thres_dict):
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
    result=temp.copy()
    result[2] = temp[2][temp[1]>[thres_dict[temp[0][i]] for i in range(len(temp[0]))]]
    result[1] = temp[1][temp[1]>[thres_dict[temp[0][i]] for i in range(len(temp[0]))]]
    result[0] = temp[0][temp[1]>[thres_dict[temp[0][i]] for i in range(len(temp[0]))]]
    return result[0],result[1],result[2]
    
def add_bounding(img_path,quyu_path): #add bounding on img file
    origin = cv2.imread(img_path)
    quyu = cv2.imread(quyu_path)
    quyu = cv2.resize(quyu, (origin.shape[1], origin.shape[0]))
    _, th = cv2.threshold(cv2.cvtColor(quyu.copy(), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    th = 255 - th
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_img = cv2.drawContours(origin.copy(), contours, -1, (0, 0, 255), 3)
    b, g, r = cv2.split(final_img)
    final_img = cv2.merge([r, g, b])
    plt.figure(figsize=(19.2, 10.8))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(final_img)
    plt.savefig(img_path)
    plt.close()
    
def draw_bounding(img,quyu_path): #draw bounding on img, diffierent from above
    quyu = cv2.imread(quyu_path)
    quyu = cv2.resize(quyu, (img.shape[1], img.shape[0]))
    _, th = cv2.threshold(cv2.cvtColor(quyu.copy(), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    th = 255 - th
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_img = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
    return final_img
    
def find_max_ios_box(test_box,box_list): 
    index=None
    max_ios=0.2
    for i in range(len(box_list)):
        box1=test_box
        box2=box_list[i]
        local_ios=retina_IOS(box1,box2)
        if local_ios>max_ios:
            max_ios=local_ios
            index=i
    return index,max_ios
    
def retina_IOS(rectA,rectB):
    W = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
    H = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])
    if W <= 0 or H <= 0:
        return 0;
    SA = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
    SB = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
    min_S=min(SA,SB)
    cross = W * H
    return cross/min_S
    
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

