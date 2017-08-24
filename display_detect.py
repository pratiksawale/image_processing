import sys
import os
print sys.path
sys.path.append('/mnt/liquid')
import cv2
import numpy as np
import time
from fabric.api import sudo, env, settings, cd
import fabric
import logging
import pygame.camera
import qrtools




usrname="root"
passwd="nvidia123"
ip="10.24.212.231"

def execute_with_fabric(username_of_system, password_of_system, ip, command,dir=None):
    logging.debug("Executing with fabric : {}".format(command))
    with settings(user=username_of_system, password=password_of_system, host_string=ip, warn_only=True):
        env.output_prefix=False
        if dir != None:
            with cd(dir):
                p=sudo(command,pty=False)
        else:
            p=sudo(command,pty=False)
    return_code=p.return_code
    if return_code !=0:
        logging.error("fabric Returned : {} for command : {}".format(return_code,command))
    else:
        logging.debug("fabric Returned : {} for command : {}".format(return_code,command))
    return return_code


def startx():
    return  execute_with_fabric(usrname,passwd,ip,"X &")

def kill_X ():
    return execute_with_fabric(usrname,passwd,ip,"pkill X")


def get_white_img():
    #execute_with_fabric(usrname,passwd,ip,)
    execute_with_fabric(usrname,passwd,ip,"DISPLAY=:0 xli -onroot /root/white.jpg")
    frame=camera_read("white.jpg",write=True)
    return frame

def get_black_img():
    execute_with_fabric(usrname,passwd,ip,"DISPLAY=:0 xli -onroot /root/black.jpg")
    frame = camera_read("black.jpg",write=True)
    return frame

def get_qr_img():
    execute_with_fabric(usrname,passwd,ip,"DISPLAY=:0 xli -onroot   -fullscreen /root/QR2.jpg")
    frame = camera_read("qr.jpg",write=True)
    return frame


def camera_read_old(name,write=False):
    while True:
        cap = cv2.VideoCapture(0)
        # while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    #    cv2.imshow(name, frame)
       # cv2.waitKey(0)
        if ret == True:
            if write:
                cv2.imwrite(name, frame)
            #cv2.destroyAllWindows()
            break
    cap.release()
    return frame


def camera_read(name,write=True):
    pygame.camera.init()
    cam = pygame.camera.Camera("/dev/video0", (640, 480))
    cam.start()
    img = cam.get_image()
    pygame.image.save(img, name+".bmp")
    p=cv2.imread(name+".bmp")
    if write:
        cv2.imwrite(name,p)
    os.remove(name+".bmp")
    p = cv2.imread(name,0)
    cam.stop()
    return p

def ready_simple(white_img, black_img, qr_image, row=0.0289, sigma=0.33,open_iterations=2, size=(800, 600), frame=False):
    #
    #    if frame == False:
    #        image = cv2.imread(img)
    #        image = cv2.imread('display_img.jpg')
    #    else:
    #        image = img
    # white_img = "white_frame.jpg"
    # black_img = "black_frame.jpg"
    # qr_image = "qr_frame.jpg"
    #
    # image = cv2.imread(white_img, 0)
    # image2 = cv2.imread(black_img, 0)
    # qr_image = cv2.imread(qr_image, 0)

    image=white_img
    image2=black_img
    image = image - image2
    image =np.uint8(image)
    print type(image)
    print image.shape

    # bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # bw_image = cv2.bilateralFilter(bw_image, 11, 17, 17)
    # cv2.imshow("2. Grayscale Image",bw_image)
    # cv2.waitKey()

    original_image=image.copy()
    # cv2.imshow("1. Input Image", image)
    # cv2.waitKey()
    bw_image = image
    cv2.imwrite("diff.jpg", np.array(bw_image))

    #    original_image = cv2.resize(image,size)
    #    qr_image = cv2.resize(qr_image,size)
    #    image=original_image
    #    cv2.imshow("2. Resized Image",qr_image)
    #    cv2.waitKey()


    #    bw_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #    bw_image = cv2.bilateralFilter(bw_image, 11, 17, 17)
    #    cv2.imshow("2. Grayscale Image",bw_image)
    #    cv2.waitKey()

    #
    #    blank_image = np.zeros((size[1],size[0]), np.uint8)
    #    cv2.imshow("2. Blank Image",blank_image)
    #    cv2.waitKey()
    #     #cv2.destroyAllWindows()

    #    display_is_black=True
    #    if display_is_black:
    #        imagem = cv2.bitwise_not(bw_image)
    #        imagem = (255-imagem)
    #        imagem = (blank_image-bw_image)
    #        cv2.imshow("2. after deleting from noise Image",imagem)
    #        cv2.waitKey(0)

    #    bw_image = imagem
    #
    #    bw_image = imagem
    #    bw_image = (255-bw_image)
    # cv2.imshow('Inverted', bw_image)
    # cv2.waitKey(0)
    #    dilate_iter=1
    #    kernel = np.ones((5,5),np.uint8)
    #    bw_image = cv2.dilate(bw_image,kernel,iterations = dilate_iter)
    #    cv2.imshow('dialated', bw_image)
    #    cv2.waitKey(0)
    #
    #    open_iterations=50
    kernel = np.ones((5, 5), np.uint8)
    bw_image = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    # cv2.imshow('opening', bw_image)
    # cv2.waitKey(0)

    # bw_image = cv2.bilateralFilter(bw_image, 11, 17, 17)
    #blurred = cv2.GaussianBlur(bw_image, (3, 3), 0)
    cv2.imwrite("blurred.jpg", np.array(bw_image))

    v = np.median(bw_image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(bw_image, 30, 150, apertureSize=3)
    cv2.imwrite("canny.jpg", np.array(edged))
    # cv2.imshow('3. Canny Edges', edged)
    # cv2.waitKey(0)

    ##alternative to canny
    #(thresh, edged) = cv2.threshold(edged.copy(), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Contours found = " + str(len(contours)))
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imwrite("contours.jpg", np.array(image))
    # cv2.imshow('5. Contours', np.array(image))
    # cv2.waitKey(0)
     #cv2.destroyAllWindows()

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    perimeters = [cv2.arcLength(sorted_contours[i], True) for i in range(len(contours))]
    pp = sorted(perimeters, reverse=True)

    c = sorted_contours[0]
    peri = row * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, peri, True)
    print len(approx)
    # approx = sorted_contours[2]
    for i in range(1, 20):
        try:
            c = sorted_contours[i]
            image = original_image.copy()
            peri = row * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, peri, True)
            print len(approx)
            if len(approx) == 4:
                break
            else:
                sorted_contours.remove(sorted_contours[i])
        except:
            pass
            return

            #    cv2.drawContours(image, [approx], -1, (255,0,0), 3)
            #    final_image=image
            #    cv2.imshow('Contours by area', image)
            #    cv2.waitKey(0)
            #     #cv2.destroyAllWindows()

    qr_contours = qr_image.copy()
    cv2.drawContours(qr_contours, [approx], -1, (255, 0, 0), 3)
    # cv2.imshow('Contours by area', qr_contours)
    # cv2.waitKey(0)
     #cv2.destroyAllWindows()

    # (br, bl, tl, tr) = approx

    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    #    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    #    cv2.imshow('Cropped by area', warp)
    #    cv2.waitKey(0)
    #     #cv2.destroyAllWindows()

    warp = cv2.warpPerspective(qr_image.copy(), M, (maxWidth, maxHeight))
    # cv2.imshow('Cropped by area', warp)
    # cv2.waitKey(0)
     #cv2.destroyAllWindows()
    cv2.imwrite("cropped.jpg",warp)

    return warp


def run():
    p=startx()
    time.sleep(2)
    white=get_white_img()
    qr=get_qr_img()
    black = get_black_img()
    p=kill_X()
    img=ready_simple(white_img=white,black_img=black,qr_image=qr,row=0.02)
    qr = qrtools.QR()
    rc=qr.decode("cropped.jpg")
    if rc:
        print "Detected Successfully."
        print qr.data




#camera_read("test.jpg",write=True)
run()
p=raw_input()
#
# a=cv2.imread("white.jpg",0)
# b=cv2.imread("black.jpg",0)
# c=cv2.imread("qr.jpg",0)
# ready_simple(white_img=a,black_img=b,qr_image=c)
