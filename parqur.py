#Trackbars sekmesinden ışığa göre objenin görünüp görünmediğini ayarlayabilirsin.
#Bazı yerlerde hangi değişken ne döndürdüğüne bakmak için print vs ekledim.Onlara takılma.
#yuvarlaklık,kaçgen olduğu,boyutu,çemberselliği,konvex olup olmadığı gibi bir sürü koşul koydum.
#Şuanda kırmızı renk elle ayarlanıyor.Fakat rangelemek istersen, 50-58 arasını commentten al ,60-61 i commentle.
#130.satırdaki if'ten sonra objenin orta koordinatları veriliyor.
#@ dasmehdix
import cv2
import numpy as np

def pixDis(a1,b1,a2,b2):
	#distance between points 
	y = b2-b1
	x = a2-a1
	return np.sqrt(x*x+y*y)


def nothing(x):
	# any operation
	pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 62, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

#shape_image = cv2.imread("export.png")
global center

while True:
	ret, frame = cap.read()
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
	
	''' lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])
	mask0 = cv2.inRange(hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red = np.array([170,50,50])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(hsv, lower_red, upper_red)
	mask = mask0+mask1 '''

	lower_red = np.array([l_h, l_s, l_v])
	upper_red = np.array([u_h, u_s, u_v])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.dilate(mask, kernel)
	mask[np.where(mask==0)] = 0
	#mask = cv2.Canny(mask,50,100)

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
			''' print(cv2.boundingRect(cnt))
			cv2.rectangle(frame,(x1-20,y1-20),(x1+w1+20,y1+h1+20),(0,255,0),2) '''
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
							''' print(gray.shape)
							cv2.imshow("ngray", gray)  '''
							rows = gray.shape[0]
							#cv2.imshow("daire",gray)
							circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,rows/8,param1=50,param2=15,minRadius=15,maxRadius=120)
							if circles is not None:
									circles = np.uint16(np.around(circles))
									for i in circles[0,:]:
										center = (i[0]+x1, i[1]+y1)	
										radius = i[2]
										#cv2.circle(frame, center, radius, (111, 23, 111), 3)
									''' if len(approx.ravel()) <= 12:
										yayDif = pixDis(approx.ravel()[0],approx.ravel()[1],approx.ravel()[4],approx.ravel()[5])
										#yaydif cemberde ilk ve 3. nokta arasındaki mesafe ref ise olması gerek uzunluğu
										referance = np.sqrt(2)*pixDis(approx.ravel()[0],approx.ravel()[1],approx.ravel()[8],approx.ravel()[9])
										if referance-2<yayDif<referance+2: '''
									if area >= oldArea:		
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
	
		


			''' if len(approx) == 3:
				cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
			elif len(approx) == 4:
				cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
			elif len(approx) == 5:
				cv2.putText(frame,"Half-Circle",(x,y),font,1,(0,255,0))    
			elif 10 < len(approx) < 20:
				cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0)) '''


	cv2.imshow("Erik Tech Labs", frame)
	cv2.imshow("Maske", mask)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()