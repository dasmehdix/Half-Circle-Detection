import os
import numpy as np 
import cv2
from skimage import feature as ft 
from sklearn.externals import joblib

cls_names = ["straight", "left", "right", "stop", "nohonk", "crosswalk", "background"]
img_label = {"straight": 0, "left": 1, "right": 2, "stop": 3, "nohonk": 4, "crosswalk": 5, "background": 6}

def preprocess_img(imgBGR, erode_dilate=True):
	"""preprocess the image for contour detection.
	Args:
		imgBGR: source image.
		erode_dilate: erode and dilate or not.
	Return:
		img_bin: a binary image (blue and red).

	"""
	rows, cols, _ = imgBGR.shape
	imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

	Bmin = np.array([88, 131, 97])
	Bmax = np.array([180, 255, 255])
	img_Bbin = cv2.inRange(imgHSV,Bmin, Bmax)
	
	''' Rmin1 = np.array([0, 43, 46])
	Rmax1 = np.array([10, 255, 255])
	img_Rbin1 = cv2.inRange(imgHSV,Rmin1, Rmax1) '''
	
	''' Rmin2 = np.array([156, 43, 46])
	Rmax2 = np.array([180, 255, 255])
	img_Rbin2 = cv2.inRange(imgHSV,Rmin2, Rmax2)
	img_Rbin = np.maximum(img_Rbin1, img_Rbin2) '''
	img_bin = img_Bbin

	if erode_dilate is True:
		kernelErosion = np.ones((3,3), np.uint8)
		kernelDilation = np.ones((3,3), np.uint8) 
		img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
		img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

	return img_bin


def contour_detect(img_bin, min_area=40, max_area=-1, wh_ratio=2.0):
	"""detect contours in a binary image.
	Args:
		img_bin: a binary image.
		min_area: the minimum area of the contours detected.
			(default: 0)
		max_area: the maximum area of the contours detected.
			(default: -1, no maximum area limitation)
		wh_ratio: the ration between the large edge and short edge.
			(default: 2.0)
	Return:
		rects: a list of rects enclosing the contours. if no contour is detected, rects=[]
	"""
	rects = []
	_, contoursa, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(contoursa) == 0:
		return rects

	max_area = (img_bin.shape[0]*img_bin.shape[1])/2 if max_area<0 else max_area
	for contour in contoursa:
		area = cv2.contourArea(contour)
		if area >= min_area and area <= max_area:
			x, y, w, h = cv2.boundingRect(contour)
			if 1.0*w/h < wh_ratio and 1.0*h/w < wh_ratio:
				rects.append([x,y,w,h])
	return rects


def draw_rects_on_img(img, rects):
	""" draw rects on an image.
	Args:
		img: an image where the rects are drawn on.
		rects: a list of rects.
	Return:
		img_rects: an image with rects.
	"""
	img_copy = img.copy()
	for rect in rects:
		x, y, w, h = rect
		cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2)
	return img_copy



def hog_extra_and_svm_class(proposal, clf, resize = (64, 64)):
	"""classify the region proposal.
	Args:
		proposal: region proposal (numpy array).
		clf: a SVM model.
		resize: resize the region proposal
			(default: (64, 64))
	Return:
		cls_prop: propabality of all classes.
	"""
	img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, resize)
	bins = 9
	cell_size = (8, 8)
	cpb = (2, 2)
	norm = "L2"
	features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size, 
						cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
	features = np.reshape(features, (1,-1))
	cls_prop = clf.predict_proba(features)
	cls_prop = cls_prop[0]
	cls_num = clf.predict(features)
	return cls_prop

def pixDis(a1,b1,a2,b2):
		#distance between points 
	y = b2-b1
	x = a2-a1
	return np.sqrt(x*x+y*y)


def nothing(x):
	# any operation
	pass

cap = cv2.VideoCapture(0)
cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
clf = joblib.load("svm_hog_classification/svm_model.pkl")


cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 62, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX
global center

while (1):
	ret, img = cap.read()
	frame = img.copy()
	img_bin = preprocess_img(img,False)
	min_area = img_bin.shape[0]*img.shape[1]/(25*25)
	rects = contour_detect(img_bin, min_area=min_area)
	img_bbx = img.copy()

	for rect in rects:
		xc = int(rect[0] + rect[2]/2)
		yc = int(rect[1] + rect[3]/2)

		size = max(rect[2], rect[3])
		x1 = max(0, int(xc-size/2))
		y1 = max(0, int(yc-size/2))
		x2 = min(cols, int(xc+size/2))
		y2 = min(rows, int(yc+size/2))
		proposal = img[y1:y2, x1:x2]
		cls_prop = hog_extra_and_svm_class(proposal, clf)
		cls_prop = np.round(cls_prop, 2)
		cls_num = np.argmax(cls_prop)
		cls_name = cls_names[cls_num]
		if cls_name == "left" or cls_name == "right":
			cv2.rectangle(frame,(rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0,255,0), 2)
			cv2.putText(frame, cls_name, (rect[0], rect[1]), 1, 1, (0,0,255),2)
			area_traffic_sign = rect[2]*rect[3]
	
	frame = cv2.GaussianBlur(frame,(5,5),0)
	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	l_h = cv2.getTrackbarPos("L-H", "Trackbars")
	l_s = cv2.getTrackbarPos("L-S", "Trackbars")
	l_v = cv2.getTrackbarPos("L-V", "Trackbars")
	u_h = cv2.getTrackbarPos("U-H", "Trackbars")
	u_s = cv2.getTrackbarPos("U-S", "Trackbars")
	u_v = cv2.getTrackbarPos("U-V", "Trackbars")

	# lower mask (0-10)

	lower_red = np.array([l_h, l_s, l_v])
	upper_red = np.array([u_h, u_s, u_v])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.dilate(mask, kernel)
	mask[np.where(mask==0)] = 0
	

	# Contours detection
	if int(cv2.__version__[0]) > 3:
		# Opencv 4.x.x
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	else:
		# Opencv 3.x.x
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	oldArea = 0
	oldCnt = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
		diameter = np.sqrt(4*area/np.pi)
		x = approx.ravel()[0]
		y = approx.ravel()[1]
		convvex = cv2.isContourConvex(cnt)
		#print(len(cnt))
		if cv2.arcLength(cnt,True) != 0:
			circularity = (4*np.pi*area)/((cv2.arcLength(cnt,True))*(cv2.arcLength(cnt,True)))
	
		if area > 500 and 0.5 <=float(circularity) <= 0.7 and 4< len(approx) <= 7 and convvex == False:
			#roundness = (4*area)/(np.pi*diameter*diameter)
			#print(float(roundness))
			x1, y1 ,w1, h1 = cv2.boundingRect(cnt)
			x2 = x1 + w1 
			y2 = y1 + h1 
			if x1<0:
				x1=1
			if y1<0:
				y1=1
			if x2>639:
				x2 = 639
			if y2>479:
				y2=479
			gray = gray[(y1):(y2),(x1):(x2)]	
			if gray.shape[0] != 0:
				if gray.shape[1] != 0:
					if y2>y1:
						if x2>x1:
							rrows = gray.shape[0]
							#cv2.imshow("daire",gray)
							circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,rrows/8,param1=50,param2=15,minRadius=15,maxRadius=120)
							if circles is not None:
									circles = np.uint16(np.around(circles))
									for i in circles[0,:]:
										center = (i[0]+x1, i[1]+y1)	
										radius = i[2]
										#cv2.circle(frame, center, radius, (111, 23, 111), 3)
									if area >= oldArea:
										area_circle = cv2.contourArea(cnt)
										M = cv2.moments(cnt)
										cX = int(M["m10"] / M["m00"])
										cY = int(M["m01"] / M["m00"])
										if abs(center[0]-cX) < 40:
											#OBJENIN DAIRE OLDUGUNA KARAR VERDIGIMIZ YER BURASI cX,cY objenin tam orta noktasıdır.
											cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
											cv2.line(frame,(cX,cY),(cX,cY-60),(0,255,0),thickness=2)
											cv2.line(frame,(cX,cY),(cX+60,cY),(255,0,0),thickness=2)
											cv2.line(frame,(cX,cY),(cX-30,cY+30),(0,0,255),thickness=2)
											cv2.putText(frame,"Yarim Daire Bulundu",(x,y),font,1,(100,244,237),thickness=2)
											oldArea = area
											oldCnt = cnt

	try:
		if area_circle >= 0 and area_traffic_sign >= 0:
			if area_circle >= area_traffic_sign:
				cv2.putText(frame,"Daire Daha yakinda",(30,30),font,1,(100,244,237),thickness=2)
			else:
				cv2.putText(frame,"Trafik Daha Yakinda",(30,30),font,1,(254,111,111),thickness=2)
		else:
			cv2.putText(frame,"Tek veya Hiç Bulgu",(30,400),font,1,(1,3,237),thickness=2)
	except:
		cv2.putText(frame,"Tek veya Hiç Bulgu",(30,400),font,1,(1,3,237),thickness=2)
	
	area_circle = 0
	area_traffic_sign = 0



	cv2.imshow("detect result", frame)
	cv2.imshow("maske",mask)
	cv2.waitKey(20)