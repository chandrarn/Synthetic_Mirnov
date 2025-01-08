#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 18:12:11 2025
    Convert point cloud to ( (x0,y0,z0), norm)
@author: rian
"""

from header import np, Mirnov, save_sensors,plt,cm

def conv_sensor(sensor_file):
    
    dat=np.loadtxt(sensor_file,skiprows=2)
    
    radius = []
    norm_v = []
    x0 = []
    fn_norm = lambda vec: np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    sensors=[]
    for ind,i in enumerate(np.arange(0,len(dat),200 )):
        
        pt_1 = dat[i,1:].copy()
        pt_2 = dat[i+int(33),1:] # opposite point
        pt_3 = dat[i+int(16),1:].copy() # right angle point
        
        pt_0 = (pt_2-pt_1)/2 + pt_1 # center point
        
        pt_1-=pt_0
        pt_3-=pt_0
        
        x1,y1,z1=pt_1
        x2,y2,z2=pt_3
        
        norm = np.array([y1*z2-z1*y2,z1*x2-x1*z2,x1*y2-y1*x2])
        norm /= fn_norm(norm)
        
        # return dat, pt_1,pt_2,pt_3,pt_0,norm
        radius.append(fn_norm(pt_1))
        norm_v.append(norm)
        x0.append(pt_0)
        sensors.append(Mirnov(pt_0, norm, 'Mirnov_%d'%ind))
        '''
        fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
        ax.scatter(dat[i+np.arange(200),1],dat[i+np.arange(200),2],
                   dat[i+np.arange(200),3],'*')
        ax.scatter(*pt_0,'k*',s=20)
        ax.plot([pt_0[0],pt_0[0]+pt_1[0]],[pt_0[1],pt_0[1]+pt_1[1]],\
                [pt_0[2],pt_0[2]+pt_1[2]],)
        ax.plot([pt_0[0],pt_0[0]+pt_3[0]],[pt_0[1],pt_0[1]+pt_3[1]],\
                [pt_0[2],pt_0[2]+pt_3[2]],)
        ax.plot([pt_0[0],pt_0[0]+norm[0]*.1],[pt_0[1],pt_0[1]+norm[1]*.1],\
                [pt_0[2],pt_0[2]+norm[2]*.1],)    
        '''
    
        #plt.show()
        #break
    
    fig,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    for ind,x0_ in enumerate(x0): 
        if not(x0_[0]>0 and x0_[1]<-1):continue
        ax.plot(*x0_,'*k')
        
        orient = np.abs(np.dot(norm_v[ind],[0,0,1]))
        ax.plot([x0_[0],x0_[0]+norm_v[ind][0]*.1],[x0_[1],x0_[1]+norm_v[ind][1]*.1],\
                [x0_[2],x0_[2]+norm_v[ind][2]*.1],c=cm.get_cmap('bwr')(orient))
        
        
    plt.grid()
    
    plt.show()
    
    save_sensors(sensors)
    return sensors,dat, radius,norm_v,x0