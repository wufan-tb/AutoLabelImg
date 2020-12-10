import numpy as np
import random,cv2,copy,gc
import matplotlib.pyplot as plt

class box_Tracker:
    def __init__(self):
        self.miss_frames=0
        self.kalman = cv2.KalmanFilter(8, 4) # 状态包括（x,y,w,h,dx,dy,dw,dh）,前4个为观测值
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
                         [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32) # 系统测量矩阵
        self.kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = 0.05*np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], 
        [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)

    def tracker_correct(self,current_measurement):
        self.kalman.correct(current_measurement)   # 用当前测量来校正卡尔曼滤波器

    def tracker_predict(self):
        return self.kalman.predict()  #预测下一帧box位置
        
class Frames_obj_Tracker:
    '''
    视频帧追踪器，和视频相对应，包含了要对一路视频进行追踪的一些函数和数据，详细参数包含：
    Length：视频图片的长度，默认是1920; Height：视频图片的高度，默认是1080
    obj_id_dict：ID字典，存放了每个id对应的kalman追踪器和丢失帧数，用来对每个id进行追踪和预测，丢帧数达到一定值则舍弃该目标/id
    iou_threshold：追踪时判断两个box是否匹配的iou阈值
    max_miss_frame：最大丢帧数，当某个id丢帧数大于这个数，认为该id出视角范围（不再继续预测和追踪，就算重新出现了，也当成新目标处理）
    local_window：统计某个状态时，统计最近几帧的该状态，综合分析
    state_ratio:统计完最近几帧的状态后，状态出现最多的那个，其比率搞过这个数，才会认为该状态可信
    Max_Frames_length:保存最大帧数，为防止内存溢出，只保存最近一定帧的识别和追踪信息
    Started：追踪标记，初始为F,出现第一个box后为T
    Frames：帧列表，保存了视频每一帧的识别和追踪信息。一般格式为：[[classes],[scores],[boxes],[ids],[states],...],是可视化的重要数据
    '''
    def __init__(self, Length=1920, Height=1080, iou_threshold=0.5, max_miss_frame=10, local_window=10, state_ratio=0.4, Max_Frames_length=200):
        self.Length=Length;self.Heigth=Height 
        self.iou_threshold=iou_threshold
        self.max_miss_frame=max_miss_frame  
        self.local_window=local_window    
        self.state_ratio=state_ratio
        self.Max_Frames_length=Max_Frames_length
        self.Started=False
        self.id_index=0
        self.Frames=[]
        self.obj_id_dict={} #保存每个id及其对应的追踪器
        
    def IOU(self,rectA, rectB):
        W = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
        H = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])
        if W <= 0 or H <= 0:
            return 0
        SA = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
        SB = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
        cross = W * H
        return cross / (SA + SB - cross)
        
    def xyxy_to_xywh(self,box):
        x_center=(box[1]+box[3])/2.0
        y_center=(box[0]+box[2])/2.0
        delta_w=box[3]-box[1]
        delta_h=box[2]-box[0]
        return np.array([x_center,y_center,delta_w,delta_h])
        
    def xywh_to_xyxy(self,box):
        ymin=box[1]-(box[3]/2.0)
        xmin=box[0]-(box[2]/2.0)
        ymax=box[1]+(box[3]/2.0)
        xmax=box[0]+(box[2]/2.0)
        return np.array([ymin,xmin,ymax,xmax])
        
    def box_to_measurement(self,box):
        M=self.xyxy_to_xywh(box)
        return np.array([[np.float32(M[0])], [np.float32(M[1])],[np.float32(M[2])],[np.float32(M[3])]])
        
    def measurement_to_box(self,M):
        box=[]
        for i in range(4):
            box.append(int(M[i,0]))
        return self.xywh_to_xyxy(box)
        
    def frame_predict(self,last_frame):
        predict_frame=copy.deepcopy(last_frame)
        for i in range(len(last_frame[0])):
            current_measurement=self.box_to_measurement(last_frame[2][i])
            self.obj_id_dict[last_frame[3][i]].tracker_correct(current_measurement)
            temp=self.obj_id_dict[last_frame[3][i]].tracker_predict()
            predict_frame[2][i]=self.measurement_to_box(temp)
        return predict_frame
    
    def frame_match(self,predict_frame,current_frame):
        update_frame=copy.deepcopy(current_frame)
        update_frame.append(np.array([-1 for i in range(len(current_frame[0]))]))
        for j in range(len(predict_frame[0])): #先循环遍历预测框
            predict_class=predict_frame[0][j]
            predict_box=predict_frame[2][j]
            true_index=None
            max_iou=self.iou_threshold
            for k in range(len(current_frame[0])):
                current_class=current_frame[0][k]
                current_box=current_frame[2][k]
                if predict_class==current_class and (self.IOU(predict_box,current_box)>max_iou):
                    true_index=k
                    max_iou=self.IOU(predict_box,current_box)
            if true_index == None:  #预测框丢失
                id=predict_frame[3][j]
                self.obj_id_dict[id].miss_frames+=1
                if self.obj_id_dict[id].miss_frames <= self.max_miss_frame:  #丢帧不到一定次数
                    for kk in range(4):
                        if kk == 2:
                            try:
                                x,y=update_frame[kk].shape
                            except:
                                x,y=0,4
                            update_frame[kk]=np.append(update_frame[kk],predict_box)
                            update_frame[kk]=np.resize(update_frame[kk],(x+1,y))
                        elif kk==1:
                            update_frame[kk]=np.append(update_frame[kk],0)
                        else:
                            update_frame[kk]=np.append(update_frame[kk],predict_frame[kk][j])
                else:   #丢帧到了一定次数
                    del self.obj_id_dict[id]
                    gc.collect()
            else:    #预测框匹配成功
                id=predict_frame[3][j]
                self.obj_id_dict[id].miss_frames=0
                update_frame[3][true_index]=id
                
        for j in range(len(update_frame[0])):  #循环遍历识别框，处理上轮循环未匹配的新目标
            if update_frame[3][j]==-1:   #定位新目标
                update_frame[3][j]=self.id_index
                self.obj_id_dict[self.id_index] = box_Tracker()
                current_measurement=self.box_to_measurement(update_frame[2][j])
                for k in range(15):   #新目标卡尔曼滤波校正15次
                    self.obj_id_dict[self.id_index].tracker_correct(current_measurement)
                    temp=self.obj_id_dict[self.id_index].tracker_predict()
                self.id_index += 1
                
        return update_frame
    
    def tracker_frames_update(self,current_frame):
        #append ID
        if len(self.Frames)>=self.Max_Frames_length:
            self.Frames.remove(self.Frames[0])
        if not self.Started:
            self.Frames.append(current_frame)
            ID=[]
            for j in range(len(self.Frames[-1][0])):
                ID.append(self.id_index)
                self.obj_id_dict[self.id_index] = box_Tracker()
                current_measurement=self.box_to_measurement(current_frame[2][j])
                for k in range(15):
                    self.obj_id_dict[self.id_index].tracker_correct(current_measurement)
                    temp=self.obj_id_dict[self.id_index].tracker_predict()
                self.id_index += 1
                self.Started=True
            self.Frames[-1].append(np.array(ID))
        else:
            predict_frame=self.frame_predict(self.Frames[-1])
            temp=self.frame_match(predict_frame,current_frame)
            self.Frames.append(temp)
        #append exist state
        exist_ditc={}
        for i in range(max(0-len(self.Frames),-1-self.local_window),-1):
            if len(self.Frames[i])<0:
                continue
            else:
                for j in range(len(self.Frames[i][0])):
                    key=self.Frames[i][3][j]
                    if key not in exist_ditc.keys():
                        exist_ditc[key]=self.Frames[i][1][j]
                    else:
                        exist_ditc[key]+=self.Frames[i][1][j]

        exist_state=[]
        for item in self.Frames[-1][3]:
            if item in exist_ditc.keys():
                temp=exist_ditc[item]/self.local_window
                exist_state.append(temp)
            else:
                exist_state.append(0)
        self.Frames[-1].append(np.array(exist_state)>self.state_ratio)
        
