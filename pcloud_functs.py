#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:41:19 2021

@author: root
"""


import numpy as np
import open3d 



def vol2inds(vol):
    return np.stack(np.where(vol==1),1)

def inds2vol(Loc,volshape):
    vol = np.zeros(volshape,'bool')
    for ipt in range(Loc.shape[0]):
        vol[Loc[ipt,0],Loc[ipt,1],Loc[ipt,2]] = 1
    return vol

def lowerResolution(Loc):
    
    return np.unique(np.floor(Loc/2).astype('int'),axis=0)

def dilate_Loc(lrGT): 

# %% Move from current resolution one level down and then one up
# %% In the UP step enforce at each cube all 8 possible patterns  
# %% input: sBBi, the sorted PC, by unique

    # quotBB = np.floor(sBBi/2).astype('int')                    #% size is (nBBx3)
    # Points_parent,iC = np.unique(quotBB,return_inverse=True,axis=0)  #   % size of iC is (nBBx3)

    PatEl = np.array( [[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]])
                       

    
    BlockiM = np.zeros([0,3],'int')
    for iloc in range(8):    # % size(PatEl,1) = 8
        iv3 = PatEl[iloc,:]
        Blocki = lrGT*2+iv3
        BlockiM = np.concatenate((BlockiM,Blocki),0) 

    LocM = np.unique(BlockiM,axis=0)
    return LocM


def pcread(ply_path):
    pcd = open3d.io.read_point_cloud(ply_path )
    GT = np.asarray(pcd.points,'int16')   
    return GT


def pcfshow(ply_path):
    pcd = open3d.io.read_point_cloud(ply_path )
    open3d.visualization.draw_geometries([pcd]) #, zoom=0.3412,


def pcshow(Location):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.cpu.pybind.utility.Vector3dVector(Location)
    open3d.visualization.draw_geometries([pcd]) 

def collect_blocks(inLoc,block_shape,max_num_blocks):               
                          
    maxes = np.max(inLoc,0)
    mins = np.min(inLoc,0)
    ws = np.array(block_shape,'int8')
    ns = np.ceil(maxes/ws).astype('int16')
    
    minis = np.ceil(mins/ws-1).astype('int8')
    
    
    blocks = []
    done=0
    for ix in range(minis[0],ns[0]):
        for iy in range(minis[1],ns[1]):
            for iz in range(minis[2],ns[2]):
                #print([ix,iy,iz])
                xlow = ix*ws[0]
                ylow = iy*ws[1]
                zlow = iz*ws[2]
                xhigh = xlow+ws[0]
                yhigh = ylow+ws[1]
                zhigh = zlow+ws[2]
                cubinds = (inLoc[:,0]<xhigh) * (inLoc[:,0]>=xlow) * (inLoc[:,1]<yhigh) * (inLoc[:,1]>=ylow) *(inLoc[:,2]<zhigh) * (inLoc[:,2]>=zlow)
                if np.any(cubinds):
                    cLocs = inLoc[cubinds,:]-np.array([xlow,ylow,zlow])      
                    blocks.append(cLocs)
                    print('num cubes:'+str(len(blocks)) + 'cLocs.shape[0]:'+str(cLocs.shape[0]))
                    print(str([ix,iy,iz]) +'/'+str(ns) )
                    if(len(blocks)==max_num_blocks):
                        done=1
                        break
            if(done):
                break
            
        if(done):
            break
    return blocks

def collect_counts(inLoc,block_shape,max_num_ctxs,curr_loc_inds):               
                          
    maxes = np.max(inLoc,0)
    mins = np.min(inLoc,0)
    ws = np.array(block_shape,'int8')
    ns = np.ceil(maxes/ws).astype('int16')
    
    minis = np.ceil(mins/ws-1).astype('int8')
    
    n_pixels = np.prod(block_shape)-1
    
    ctxs = -1*np.ones((max_num_ctxs,n_pixels))
    n_ctxs = 0
    counts = np.zeros((max_num_ctxs,2),'int32')
    done=0
    ctx = np.zeros(n_pixels,'bool')
    for ix in range(minis[0],ns[0]):
        for iy in range(minis[1],ns[1]):
            for iz in range(minis[2],ns[2]):
                #print([ix,iy,iz])
                xlow = ix*ws[0]
                ylow = iy*ws[1]
                zlow = iz*ws[2]
                xhigh = xlow+ws[0]
                yhigh = ylow+ws[1]
                zhigh = zlow+ws[2]
                cubinds = (inLoc[:,0]<xhigh) * (inLoc[:,0]>=xlow) * (inLoc[:,1]<yhigh) * (inLoc[:,1]>=ylow) *(inLoc[:,2]<zhigh) * (inLoc[:,2]>=zlow)
                if np.any(cubinds):
                    cLocs = inLoc[cubinds,:]-np.array([xlow,ylow,zlow])      
                    block = np.zeros(block_shape,'bool')
                    for ip in range(cLocs.shape[0]):                       
                        block[ cLocs[ip,0], cLocs[ip,1], cLocs[ip,2]] = 1
                    # print('num cubes:'+str(len(blocks)) + 'cLocs.shape[0]:'+str(cLocs.shape[0]))
                    
                    symb = block[curr_loc_inds[0],curr_loc_inds[1],curr_loc_inds[2]]
                    
                    ibit = 0
                    for ibx in range(block_shape[0]):
                        for iby in range(block_shape[1]):
                            for ibz in range(block_shape[2]):
                                if not([ibx,iby,ibz] == curr_loc_inds):
                                    ctx[ibit] = bool(block[ibx,iby,ibz])
                                    ibit+=1
                 
                    ctx_where = np.where((ctxs[0:n_ctxs,:] == ctx).all(axis=1))
                    if len(ctx_where[0])>0: ## ctx is one of the previously encountered ctxs
                        ctx_ind = ctx_where[0][0]                      
                    else: # a new ctx is encountered
                        ctx_ind = n_ctxs
                        ctxs[ctx_ind,:] = ctx
                        n_ctxs+=1
                        
                    counts[ctx_ind,int(symb)]+=1   
                    
                    print(str([ix,iy,iz]) +'/'+str(ns) )
                    print('n_ctxs'+str(n_ctxs) )
                    if(n_ctxs==max_num_ctxs):
                        done=1
                        break
            if(done):
                break
            
        if(done):
            break
    return counts,ctxs


def collect_counts2(inLoc,block_shape,max_num_ctxs,curr_loc_inds):               
                          
    maxes = np.max(inLoc,0)
    mins = np.min(inLoc,0)
    ws = np.array(block_shape,'int8')
    # ns = np.ceil(maxes/ws).astype('int16')
    
    #minis = np.ceil(mins/ws-1).astype('int8')
    n_pixels = np.prod(block_shape)-1
    ctx = np.zeros(int(n_pixels),'bool')
    n_pixels = np.prod(block_shape)-1
    
    ctxs = -1*np.ones((int(max_num_ctxs),int(n_pixels)))
    n_ctxs = 0
    counts = np.zeros((int(max_num_ctxs),2),'int32')
    done=0
    
    max_lows = maxes-ws
    min_lows = mins-ws
    for xlow in range(int(min_lows[0]),int(max_lows[0])):
        for ylow in range(int(min_lows[1]),int(max_lows[1])):
            for zlow in range(int(min_lows[2]),int(max_lows[2])):
                        
                xhigh = xlow+ws[0]
                yhigh = ylow+ws[1]
                zhigh = zlow+ws[2]
                cubinds = (inLoc[:,0]<xhigh) * (inLoc[:,0]>=xlow) * (inLoc[:,1]<yhigh) * (inLoc[:,1]>=ylow) *(inLoc[:,2]<zhigh) * (inLoc[:,2]>=zlow)
                if True:#np.any(cubinds):
                    cLocs = inLoc[cubinds,:]-np.array([xlow,ylow,zlow])      
                    block = np.zeros(block_shape,'bool')
                    for ip in range(cLocs.shape[0]):                       
                        block[ cLocs[ip,0], cLocs[ip,1], cLocs[ip,2]] = 144
                            # print('num cubes:'+str(len(blocks)) + 'cLocs.shape[0]:'+str(cLocs.shape[0]))
                            
                    symb = block[curr_loc_inds[0],curr_loc_inds[1],curr_loc_inds[2]]
                        
                    ibit = 0
                    for ibx in range(block_shape[0]):
                        for iby in range(block_shape[1]):
                            for ibz in range(block_shape[2]):
                                if not([ibx,iby,ibz] == curr_loc_inds):
                                    ctx[ibit] = bool(block[ibx,iby,ibz])
                                    ibit+=1
                 
                    ctx_where = np.where((ctxs[0:n_ctxs,:] == ctx).all(axis=1))
                    if len(ctx_where[0])>0: ## ctx is one of the previously encountered ctxs
                        ctx_ind = ctx_where[0][0]        
                        counts[ctx_ind,int(symb)]+=1  
                    else: # a new ctx is encountered
                        ctx_ind = n_ctxs
                        ctxs[ctx_ind,:] = ctx
                        n_ctxs+=1
                        counts[ctx_ind,int(symb)]+=1  
                        if(n_ctxs==max_num_ctxs):
                            done=1
                            break    
                     
                    


                    print(str([xlow,ylow,zlow]) +'/'+str(max_lows) )
                    print('n_ctxs'+str(n_ctxs) )     
                    
            if(done):
                break
            
        if(done):
            break
    return counts,ctxs
    
    
    
    
    
    




def ctxbits2block(ctx,block_shape,symb_inds,symb):
    
    block = np.zeros(block_shape,'bool')
    ibit = 0
    for ibx in range(block_shape[0]):
        for iby in range(block_shape[1]):
            for ibz in range(block_shape[2]):
                if not([ibx,iby,ibz] == symb_inds):
                    block[ibx,iby,ibz] = ctx[ibit]
                    ibit+=1
                else:
                    block[ibx,iby,ibz] = symb
                    
                    
    return block



# def deneme(inLoc,block_shape,max_num_ctxs,curr_loc_inds):               
                          
#     maxes = np.max(inLoc,0)
#     mins = np.min(inLoc,0)
#     ws = np.array(block_shape,'int8')
#     return maxes,mins,ws








