"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   th_func.py
"""

def threshold(src,thresh,maxval):
    
    
    img = src.copy()
    rows, cols = img.shape[:2]
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] < thresh:
                img[i,j] = 0
            else:
                img[i,j] = maxval
    
    return thresh, img
