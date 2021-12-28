#!python3
# Author: Moies Garin
# Created: 8/10/2021
import cv2
import numpy as np
from matplotlib import pyplot as plt
class BBox:
    """ A class holding a bounding box 
    """
    def __init__(self,left=0,top=0,right=0,bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


    def move(self,dx,dy):
        """ Move the bbox some increment """
        self.left = self.left + dx
        self.right = self.right + dx
        self.top = self.top + dy
        self.bottom = self.bottom + dy
        
    def make_square(self):
        """ Makes sure the bbox is square. Enlarge
        if needed """
        w = self.width
        h = self.height
        
        if w<h:
            self.width = h
        elif h<w:
            self.height = w
        return w
            

    @property
    def center(self):
        x = 0.5*(self.left + self.right)
        y = 0.5*(self.top + self.bottom)
        return (x,y)
    
    @center.setter
    def center(self,pos):
        xc,yc = self.center
        x,y = pos
        dx = x-xc
        dy = y-yc
        self.move(dx,dy)
        
        
    @property
    def size(self):
        return (self.width, self.height)
    
    @property
    def width(self):
        """ Return width of bbox """
        return self.right - self.left
    
    @width.setter
    def width(self,value):
        """ Set width of the bbox withoout
        changing the center position """
        x,y = self.center
        self.left = x - 0.5*value
        self.right = x + 0.5*value
        
    @property
    def height(self):
        """ Return width of bbox """
        return self.bottom - self.top
    
    @height.setter
    def height(self,value):
        """ Set width of the bbox withoout
        changing the center position """
        x,y = self.center
        self.top = y - 0.5*value
        self.bottom = y + 0.5*value
        
    
    @property
    def pt1(self):
        return (self.left, self.top)
    
    @property
    def pt2(self):
        return (self.right, self.bottom)
    
    
    def draw(self,im):
        """ Draw the bbox in the image im"""
        pt1 = tuple(int(x) for x in self.pt1)
        pt2 = tuple(int(x) for x in self.pt2)
        cv2.rectangle(im, 
                      pt1, 
                      pt2, 
                      color = (200,0,0),
                      thickness = 2)
        
def shift_rotation(x,y,z,n,im):
    # get the size of the image
    h,w,_ = im.shape
    imsize = (w,h)
    # Define a bounding matching an egg.
    box = BBox()
    box.width = z+25
    box.height = n+25
    box.center = (x+z*0.5,y+n*0.5)
    
    # Make sure the bbox is quare.
    box.make_square()
    # Augmentation parameters.
    # Essentialy generate random rotation and shift.
    ###################################################
    theta = np.random.uniform(-180,180) # rotation
    # maximum position shift a 5% fo the size.
    dx = box.width*.05 #!!!!!!!!!!
    dy = box.height*.05 #!!!!!!!!!!
    sx = np.random.uniform(-dx,dx) # shift pixels in x
    sy = np.random.uniform(-dy,dy) # shift pixels in y
    
    # construct the rotation around a center affine transform.
    M1 = cv2.getRotationMatrix2D(box.center, theta, 1)
    # print(M1)
    # Modify the matrix to include additional shifting.
    M1[0,2] += sx
    M1[1,2] += sy

    # Rotate + translate the image using the 
    # affine transform.
    im2 = cv2.warpAffine(im, M1, imsize)


    # Cut the image from the transformed image.
    ################################################
    l = int(box.left)
    r = int(box.right)
    t = int(box.top)
    b = int(box.bottom)
    subimg = im2[t:b, l:r, :].copy()
    subimg = cv2.resize(subimg,(150,150))


    # Apply extra augmentation after cutting
    # the image.
    ###########################################

    # Randomly flip the image vertically
    if np.random.choice((True,False)):
        subimg = subimg[::-1,:,:]
        
    # Randomly flip the image horizontally
    if np.random.choice((True,False)):
        subimg = subimg[:,::-1,:]
        
    
    # Other possible augmentation/postprocessing procedures.
    # - Change brightness & contrast?
    # - shift colors?
    # - Change to B&W
    # - Some kind of image deformation?
        


    # Show the results.
    ###############################################

    # Show the original image with the bbox.
    # Show the rotated image and the bbox
    #box.draw(im)
    #cv2.imshow("img",im)
    #cv2.waitKey(0)

    # Show the rotated image and the bbox
    #box.draw(im2)
    #cv2.imshow("img2",im2)
    #cv2.waitKey(0)

    # Show the final "augmented" image of the egg.
    #cv2.imshow("subimg", subimg)

    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image

    return subimg

def black_pixels(img):
    image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = cv2.countNonZero(image)
    total=image.size
    blacks=total-count
    percentatge=(blacks/total)*100
    if percentatge>0:
        return True
    else:
        return False

if __name__=="__main__":
    im=cv2.imread(r'C:\Users\Erick\Desktop\TFG\Documentos\flash\IMG_20210309_095807.jpg')
    im= cv2.resize(im, None,fx=.25,fy=.25)  
    cv2.imshow("Imagen",im)
    cv2.waitKey(0)
    shift_rotation(396,7,123,130,im)
