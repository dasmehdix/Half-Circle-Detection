#Trackbars sekmesinden ışığa göre objenin görünüp görünmediğini ayarlayabilirsin.
#Bazı yerlerde hangi değişken ne döndürdüğüne bakmak için print vs ekledim.Onlara takılma.
#yuvarlaklık,kaçgen olduğu,boyutu,çemberselliği,konvex olup olmadığı gibi bir sürü koşul koydum.
#Şuanda kırmızı renk elle ayarlanıyor.Fakat rangelemek istersen, 50-58 arasını commentten al ,60-61 i commentle.
#130.satırdaki if'ten sonra objenin orta koordinatları veriliyor.
#@ dasmehdix
import cv2
import numpy as np
from scipy.stats import itemfreq

def pixDis(a1,b1,a2,b2):
	#distance between points 
	y = b2-b1
	x = a2-a1
	return np.sqrt(x*x+y*y)

def get_dominant_color(image, n_colors):
	pixels = np.float32(image).reshape((-1, 3))
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	flags, labels, centroids = cv2.kmeans(
		pixels, n_colors, None, criteria, 10, flags)
	palette = np.uint8(centroids)
	return palette[np.argmax(itemfreq(labels)[:, -1])]


clicked = False
def onMouse(event, x, y, flags, param):
	global clicked
	if event == cv2.EVENT_LBUTTONUP:
		clicked = True

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
flag = 1

global center

while True:
	ret, frame = cap.read()
	frame = cv2.GaussianBlur(frame,(5,5),0)
	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if flag == 0:
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
												cv2.imshow("Erik Tech Labs", frame)
												cv2.imshow("Maske", mask)
	elif flag == 1:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = cv2.medianBlur(gray, 37)
		circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
								1, 50, param1=120, param2=40)

		if not circles is None:
			circles = np.uint16(np.around(circles))
			max_r, max_i = 0, 0
			for i in range(len(circles[:, :, 2][0])):
				if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
					max_i = i
					max_r = circles[:, :, 2][0][i]
			x, y, r = circles[:, :, :][0][max_i]
			if y > r and x > r:
				square = frame[y-r:y+r, x-r:x+r]

				dominant_color = get_dominant_color(square, 2)
				if dominant_color[2] > 100:
					print("STOP")
				elif dominant_color[0] > 80:
					zone_0 = square[square.shape[0]*3//8:square.shape[0]
									* 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
					cv2.imshow('Zone0', zone_0)
					zone_0_color = get_dominant_color(zone_0, 1)

					zone_1 = square[square.shape[0]*1//8:square.shape[0]
									* 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
					cv2.imshow('Zone1', zone_1)
					zone_1_color = get_dominant_color(zone_1, 1)

					zone_2 = square[square.shape[0]*3//8:square.shape[0]
									* 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
					cv2.imshow('Zone2', zone_2)
					zone_2_color = get_dominant_color(zone_2, 1)

					if zone_1_color[2] < 60:
						if sum(zone_0_color) > sum(zone_2_color):
							print("LEFT")
						else:
							print("RIGHT")
					else:
						if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
							print("FORWARD")
						elif sum(zone_0_color) > sum(zone_2_color):
							print("FORWARD AND LEFT")
						else:
							print("FORWARD AND RIGHT")
				else:
					print("N/A")

			for i in circles[0, :]:
				cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
				cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)


		cv2.imshow('camera', frame)
			


		''' if len(approx) == 3:
			cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
		elif len(approx) == 4:
			cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
		elif len(approx) == 5:
			cv2.putText(frame,"Half-Circle",(x,y),font,1,(0,255,0))    
		elif 10 < len(approx) < 20:
			cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0)) '''

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()