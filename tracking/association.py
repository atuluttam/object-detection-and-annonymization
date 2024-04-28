import numpy as np
from scipy.spatial import distance as dist


class Association:
    def __init__(self):
        self.low = np.s_[...,:2]
        self.high = np.s_[...,2:]

    # generates IoU matrix between A & B.
    # every value of A is associated with each value of B with IoU score
    # eg A.shape = (3,4) B.shape =(2,4)
    def iouMatrix(self,A,B):
        A = A[:,None]
        B = B[None]
        A,B = A.copy(),B.copy()
        assert (A.shape[-1] ==4 and B.shape[-1]==4)
        A[self.high] += 1; B[self.high] += 1
        

        intrs = (np.maximum(0,np.minimum(A[self.high],B[self.high])
                            - np.maximum(A[self.low],B[self.low]))).prod(-1)
        
        return intrs / ((A[self.high]-A[self.low]).prod(-1)+\
                        (B[self.high]-B[self.low]).prod(-1)-intrs)

    def centroidMatrix(self,A,B):
        A = np.array([(A[:,0] + A[:,2])/2, (A[:,1] + A[:,3])/2])
        A = A.T
        B = np.array([(B[:,0] + B[:,2])/2, (B[:,1] + B[:,3])/2])
        B = B.T
        D = dist.cdist(A,B)
        return D



    