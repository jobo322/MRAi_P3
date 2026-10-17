import os, gc
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
from .model_output import single_output
#####################################################################
def SKY(US,axis):
    MAX = np.max(US,axis=axis); MIN = np.min(US,axis=axis)
    return np.where(np.abs(MAX)>=np.abs(MIN),MAX,MIN)
#####################################################################
def Para_split_2D(S_Pad,S_Shape,S_Size,Cross_Size,S_data):
    for ss in range(S_Size):
        i = ss // S_Shape[1]; j = ss % S_Shape[1]
        S_data[ss, :Cross_Size[0], 0] = S_Pad[i:i+Cross_Size[0], j+int(Cross_Size[1]/2)]
        S_data[ss, -Cross_Size[1]:, 0] = S_Pad[i+int(Cross_Size[0]/2), j:j+Cross_Size[1]]
    return S_data 

def split_2D(S,Cross_Size):
    S_Shape = S.shape; S_Size = S.size
    S_data = np.zeros((S_Size,np.sum(Cross_Size),1))
    pad_width = ((int(Cross_Size[0]/2), int(Cross_Size[0]/2-1/2)),
                 (int(Cross_Size[1]/2), int(Cross_Size[1]/2-1/2)))
    S = np.pad(S,pad_width,'wrap')
    S_data = Para_split_2D(S,S_Shape,S_Size,Cross_Size,S_data)
    Norm = np.linalg.norm(S_data, axis = (1,2))
    S_data = S_data/Norm[:,None,None]
    return S_data, Norm
#####################################################################
def split_3D_RI(S,R,Size,MODE=0):
    if MODE == 0:
        a = np.array(np.where(R))
        a = np.transpose(a)
        S_data = np.zeros((a.shape[0],np.sum(Size),1))
        pad_width = ((int(Size[0]/2), int(Size[0]/2-1/2)),
                    (int(Size[1]/2), int(Size[1]/2-1/2)),
                    (int(Size[2]/2), int(Size[2]/2-1/2)))
        S_Pad = np.pad(S,pad_width,'wrap')
        for ss in range(a.shape[0]):
            i = a[ss,0]; j = a[ss,1]; k = a[ss,2]
            S_data[ss, :Size[0], 0] = S_Pad[i:i+Size[0], j+int(Size[1]/2), k+int(Size[2]/2)]
            S_data[ss, Size[0]:-Size[2], 0] = S_Pad[i+int(Size[0]/2), j:j+Size[1], k+int(Size[2]/2)]
            S_data[ss, -Size[2]:, 0] = S_Pad[i+int(Size[0]/2), j+int(Size[1]/2), k:k+Size[2]]
            Norm = sum(S_data[ss,:,0]**2)**0.5
            S_data[ss,:,0] = S_data[ss,:,0]/Norm
        del a,pad_width,S_Pad,ss,i,j,k,Norm
        gc.collect()
    else:
        S_data = S[R][:,None]
    return S_data
#####################################################################
def Production_2D(tests_data,Dirc_2D,axis=0,Spe_The=0):
    Size = np.array([64,64],dtype=int)
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(shape=(128,1)), tf.keras.layers.TFSMLayer(Dirc_2D + "/DNN_" + str(axis), call_endpoint="serving_default")])

    RI = np.zeros_like(tests_data,dtype=bool)
    RI[np.abs(tests_data)>=Spe_The*np.std(tests_data)] = True

    pred = np.zeros((tests_data.size,1))
    tests,_ = split_2D(tests_data,Size)
    pred[RI.flatten()] = single_output(model.predict(tests[RI.flatten()], batch_size=2**16))
    pred = pred.reshape(tests_data.shape)

    del tests,model,RI
    gc.collect()
    return pred
#####################################################################
def Production_Projection(S_3D,Dirc_2D,axis,PN_The = 0,Slice_Size = 256):
    I = S_3D.shape[axis] // Slice_Size; R = S_3D.shape[axis] % Slice_Size 
    def Proj(S_3D):
        S = np.array(S_3D); S[S<np.percentile(-np.abs(S),PN_The)] = 0
        if I==0:
            F = Production_2D(SKY(S,axis),Dirc_2D,axis)
        else:
            slices = 3*[slice(None)]; slices[axis] = slice(Slice_Size)
            data = np.array(S[tuple(slices)])
            F = Production_2D(SKY(data,axis),Dirc_2D,axis)
            for i in range(1,I):
                slices = 3*[slice(None)]; slices[axis] = slice(i*Slice_Size, (i+1)*Slice_Size)
                data = np.array(S[tuple(slices)])
                F = Production_2D(SKY(data,axis),Dirc_2D,axis) if F is None else np.maximum(F,Production_2D(SKY(data,axis),Dirc_2D,axis))
            if R>0:
                slices = 3*[slice(None)]; slices[axis] = slice(-Slice_Size, None)
                data = np.array(S[tuple(slices)])
                F = np.maximum(F,Production_2D(SKY(data,axis),Dirc_2D,axis))
        del S
        return F
    F = Proj(S_3D)  
    if PN_The != 0:
        FN = Proj(-S_3D)
        F = np.where(F>FN,F,FN)
        del FN
    gc.collect()
    return F
#####################################################################
def Production_3D(tests_data,Filter,Dirc_3D,Filter_The):
    Size = np.array([64,64,64],dtype=int)
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(shape=(192,1)), tf.keras.layers.TFSMLayer(Dirc_3D+"/DNN_0", call_endpoint="serving_default")])

    RI = np.zeros_like(Filter,dtype=bool)
    RI[Filter>=Filter_The] = True

    pred = np.zeros((tests_data.size,1))
    tests = split_3D_RI(tests_data,RI,Size)
    pred[RI.flatten()] = single_output(model.predict(tests, batch_size=2**16))
    pred = pred.reshape(tests_data.shape)

    del tests,model,RI
    gc.collect()
    return pred
#####################################################################
def Multi(Proj_0, Proj_1, Proj_2):
    F0 = np.repeat(np.expand_dims(Proj_0, axis=0), Proj_1.shape[0], axis=0)
    F1 = np.repeat(np.expand_dims(Proj_1, axis=1), Proj_0.shape[0], axis=1)
    F2 = np.repeat(np.expand_dims(Proj_2, axis=2), Proj_0.shape[1], axis=2)
    return F0*F1*F2
#####################################################################
def Production_Filter(Proj_0, Proj_1, Proj_2,Support,Support_axis,Support_The):
    if Support.ndim == 1:
         F = Multi(Proj_0, Proj_1, Proj_2)
    else:
         Support[Support>=Support_The] = 1
         if Support_axis == 0: F = Multi(Support, Proj_1, Proj_2)
         if Support_axis == 1: F = Multi(Proj_0, Support, Proj_2)
         if Support_axis == 2: F = Multi(Proj_0, Proj_1, Support)
    return F
#####################################################################
def P3_3D(S_3D,Dirc_3D,Dirc_2D,Skip=[1,1,1],Support=np.zeros((1,)),Support_axis=0,Support_The=0.25,Filter_The=0.1**3,PN_The=0,Slice_Size=256):
    P3 = np.zeros_like(S_3D)
    Sup = np.zeros((1,))
    Skip = np.array(Skip,dtype='int')
    for i in range(Skip[0]):
        for j in range(Skip[1]):
            S = S_3D[i::Skip[0],j::Skip[1],:]
            Proj_0 = Production_Projection(S,Dirc_2D,0,PN_The=PN_The,Slice_Size=Slice_Size)
            Proj_1 = Production_Projection(S,Dirc_2D,1,PN_The=PN_The,Slice_Size=Slice_Size)
            Proj_2 = Production_Projection(S,Dirc_2D,2,PN_The=PN_The,Slice_Size=Slice_Size)
            if Support.ndim == 3: Sup = Production_Projection(Support[i::Skip[0],j::Skip[1],:],Dirc_2D,Support_axis)
            if Support.ndim == 2 and Support_axis == 0: Sup = Production_2D(Support[j::Skip[1],:],Dirc_2D,Support_axis)
            if Support.ndim == 2 and Support_axis == 1: Sup = Production_2D(Support[i::Skip[0],:],Dirc_2D,Support_axis)
            if Support.ndim == 2 and Support_axis == 2: Sup = Production_2D(Support[i::Skip[0],j::Skip[1]],Dirc_2D,Support_axis)              
            Filter = Production_Filter(Proj_0, Proj_1, Proj_2,Sup, Support_axis,Support_The)
            P3[i::Skip[0],j::Skip[1],:] = Production_3D(S,Filter,Dirc_3D,Filter_The)
    np.random.seed(0)
    noise = np.random.normal(size=S_3D.shape); noise *= 0.001/np.std(noise)
    P3 += noise; P3 *= np.sign(S_3D)
    del S, Proj_0, Proj_1, Proj_2, Filter, noise
    gc.collect()
    return P3
#####################################################################
def P3_2D(S_2D,Dirc_2D,Skip=[1,1],Spe_The=0.5,axis=0):
    P3 = np.zeros_like(S_2D)
    Skip = np.array(Skip,dtype='int')
    for i in range(Skip[0]):
        for j in range(Skip[1]):
            data = S_2D[i::Skip[0],j::Skip[1]]
            P3[i::Skip[0],j::Skip[1]] = Production_2D(data,Dirc_2D,axis=axis,Spe_The=Spe_The)
    
    np.random.seed(0)
    noise = np.random.normal(size=S_2D.shape); noise *= 0.001/np.std(noise)
    P3 += noise; P3 *= np.sign(S_2D)
    
    del noise, data
    gc.collect()
    return P3
