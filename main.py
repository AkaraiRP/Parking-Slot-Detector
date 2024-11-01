import sys, getopt
import cv2
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not, get_spot_from_points, NoSourceFound

FONT_SCALE = 2e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 3e-3  # Adjust for larger thickness in all images

class ParkingLotDetection:
    def __init__(self, cap, mask=None):
        self.cap = cap
        if mask is not None:
            self.connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
            self.spots = get_parking_spots_bboxes(self.connected_components)

        else:
            self.spots = []

        self.spots_status = [None for j in self.spots]
        self.diffs = [None for j in self.spots]
        self.circles = np.zeros((4, 2), np.int32)
        self.new_spot = None
        self.counter = 0

        self.extend_slots()

    def extend_slots(self, frame=None, previous_frame=None):
        if frame is not None and previous_frame is not None and self.new_spot is not None:
            x1, y1, w, h = self.new_spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            self.spots_status.append(empty_or_not(spot_crop))
            self.diffs.append(self.calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :]))

            self.get_diffs(frame, previous_frame)
            self.update_spots(frame, previous_frame)

        for x in range(0, 4):
            self.circles[x] = 0, 0
        self.new_spot = None
        self.counter = 0

        

    def calc_diff(self, im1, im2):
        return np.abs(np.mean(im1) - np.mean(im2))

    def get_diffs(self, frame, previous_frame):
        for spot_indx, spot in enumerate(self.spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            self.diffs[spot_indx] = self.calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        # print([self.diffs[j] for j in np.argsort(self.diffs)][::-1])
        

    def update_spots(self, frame, previous_frame):
        if previous_frame is None:
            arr_ = range(len(self.spots))
        else:
            arr_ = [j for j in np.argsort(self.diffs) if self.diffs[j] / np.amax(self.diffs) > 0.4]
        for spot_indx in arr_:
            spot = self.spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)

            self.spots_status[spot_indx] = spot_status

    def mousePoints(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and self.counter < 4:
            self.circles[self.counter] = x, y
            self.counter += 1
        
        elif event == cv2.EVENT_LBUTTONDOWN and self.counter == 4:
            for x in range(0, 4):
                self.circles[x] = 0, 0
            self.new_spot = None
            self.counter = 0

    def removeLastPoint(self):
        if self.counter == 0:
            del self.spots[-1]
            del self.diffs[-1]
            del self.spots_status[-1]
            
        elif self.counter == 4:
            for x in range(0, 4):
                self.circles[x] = 0, 0
            self.new_spot = None
            self.counter = 0

        else:
            last = None
            for x in range(0, 4):
                if x == 0 and self.circles[x][0] == 0 and self.circles[x][1] == 0:
                    pass
                elif x == 0 and self.circles[x][0] != 0 and self.circles[x][1] != 0:
                    last = x
                elif x != 0 and self.circles[x][0] != 0 and self.circles[x][1] != 0:
                    last = x
                else:
                    pass
            
            if last is not None:
                self.circles[last] = 0, 0
                if self.counter > 0:
                    self.counter -= 1
                

    def run(self):
        ret = True
        previous_frame = None
        frame_nmr = 0
        step = 30

        while ret:
            ret, frame = self.cap.read()
            

            if self.counter == 4:
                self.new_spot = get_spot_from_points(self.circles)
                x1, y1, w, h = self.new_spot
                points = np.array([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]])

                frame = cv2.drawContours(frame, [points], 0, (255, 0, 0), 2)

            elif self.counter != 4:
                for x in range(0, 4):
                    if self.circles[x][0] != 0 and self.circles[x][1] != 0:
                        cv2.circle(frame, (self.circles[x][0], self.circles[x][1]), 3, (255, 0, 0), cv2.FILLED)

            if frame_nmr % step == 0 and previous_frame is not None:
                self.get_diffs(frame, previous_frame)

            if frame_nmr % step == 0:
                self.update_spots(frame, previous_frame)
                try:
                    previous_frame = frame.copy()
                except AttributeError:
                    raise NoSourceFound
                

            for spot_indx, spot in enumerate(self.spots):
                spot_status = self.spots_status[spot_indx]

                x1, y1, w, h = self.spots[spot_indx]
                points = np.array([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]])

                if spot_status:
                    # frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    frame = cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)
                else:
                    # frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                    frame = cv2.drawContours(frame, [points], 0, (0, 0, 255), 2)
                # frame = cv2.circle(frame, (x1, y1), 3, (255, 0, 0), cv2.FILLED)

            
            height, width, _ = frame.shape

            font_scale = min(width, height) * FONT_SCALE
            thickness = int(np.ceil(min(width, height) * THICKNESS_SCALE))

            cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(self.spots_status)), str(len(self.spots_status))), (18, 18),
                cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), thickness=thickness, fontScale=font_scale)
            cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(self.spots_status)), str(len(self.spots_status))), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=thickness, fontScale=font_scale)
            

            cv2.namedWindow('Parking', cv2.WINDOW_NORMAL)
            cv2.imshow('Parking', frame)
            cv2.setMouseCallback('Parking', self.mousePoints)
            
            keypress = cv2.waitKey(25) & 0xFF

            if keypress == ord('q'):
                print("Exited with key 'q'")
                break

            if keypress == 13 and self.counter == 4 and self.new_spot is not None and previous_frame is not None:
                self.spots.append(self.new_spot)
                self.extend_slots(frame, previous_frame)

            if keypress == 8 and self.counter < 4 and previous_frame is not None:
                self.removeLastPoint()

            frame_nmr += 1

        self.cap.release()
        cv2.destroyAllWindows()

def main(argv):
    # video = None
    # mask = None

    video = cv2.VideoCapture('data/parking_sample_loop.mp4')
    # mask = cv2.imread('data/mask_sample.png', 0)
    mask = None

    try:
        opts, args = getopt.getopt(argv,"hv:m:",["video=","mask="])
    except getopt.GetoptError:
        print('main.py -v <video_path> -m <video_mask>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <video_path> -m <video_mask>')
            sys.exit()
        elif opt in ("-v", "--video"):
            print(arg)
            video = cv2.VideoCapture(arg)
        elif opt in ("-m", "--mask"):
            print(arg)
            mask = cv2.imread(arg, 0)

    parking = ParkingLotDetection(video, mask)
    parking.run()

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except NoSourceFound:
        print("There seems to be a problem with your source file, please make sure you are receiving a videostream via a webcam or a video file.")
   

