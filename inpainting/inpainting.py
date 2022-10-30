import cv2


cat = cv2.imread("cat.png")
cat_mask = cv2.imread("cat_mask.png",0)


filter_ = cv2.inpaint(cat,cat_mask,3,cv2.INPAINT_NS)



cv2.imshow("result",filter_)
cv2.imshow("cat",cat)
cv2.imshow("cat mask",cat_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()