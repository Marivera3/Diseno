import cv2 as cv

# This function show the pixel of the motion detected in blue rectangles
# from https://stackoverflow.com/questions/16100569/is-there-a-support-for-backgroundsubtractormog2-in-python-opencv


def showmotion(fgmask, frame, square = True):

    contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_NONE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []


    best_id = 0;

    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv.boundingRect(contour)
        if w > 10 and h > 10:
            # figure out id
            best_id = best_id + 1
            if square:
                cv.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                cv.putText(frame, str(best_id), (x,y-5), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

    return frame, (len(hierarchy) > 0)#, best_id
