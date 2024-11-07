import sys, getopt
import cv2
import numpy as np
import datetime

from util import get_parking_spots_bboxes, empty_or_not, get_spot_from_points, NoSourceFound

FONT_SCALE = 2e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 3e-3  # Adjust for larger thickness in all images

class ParkingLotDetection:
    def __init__(self, cap, mask=None):
        self.cap = cap

        # If mask file is unavailable, start with nothing for manual masking.
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
        # Basically, it gets the total pixel value of the spot, if the value passes a certain threshold,
        # that means, there's a drastic change in the slot, and not just because of daylight cycle (car for example).
        for spot_indx, spot in enumerate(self.spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            self.diffs[spot_indx] = self.calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        # print([self.diffs[j] for j in np.argsort(self.diffs)][::-1])
        # uncomment above to check for diffs

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

    def save_mask(self, mask):
        try:
            date = datetime.datetime.now()
            fname = f"saved_mask-{date.strftime('%Y-%m-%d')}"
            path = f'data/{fname}.png'

            cv2.imwrite(path, mask)
            print("Save complete!")
        except Exception as e:
            raise e

    def removeLastPoint(self):
        # Deletes whole spots
        if self.counter == 0:
            del self.spots[-1]
            del self.diffs[-1]
            del self.spots_status[-1]
            
        # I CANNOT make this work and I did not have enough time to figure it out.
        # This tries to delete the "confirmation rectangle" after drawing 4 points,
        # I don't know why this doesn't work, but it's exactly the same code snippet as just left clicking after 4 counts.
        # If someone can explain to me why this doesn't work, I'd be grateful.
        elif self.counter == 4:
            for x in range(0, 4):
                self.circles[x] = 0, 0
            self.new_spot = None
            self.counter = 0

        # Ugly implementation of getting the last "non-zero-zero (0, 0)" value in the numpy array and sets it to 0, 0.
        # I think numpy has a way to do this, but this works and also fills the criteria.
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
            
            # If we have 4 mouse clicks, arrange the click to form a rectangle via contour.
            # Contour draws in clockwise motion so it is necessary to arrange the coords before drawing.
            if self.counter == 4:
                self.new_spot = get_spot_from_points(self.circles)
                x1, y1, w, h = self.new_spot
                points = np.array([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]])

                frame = cv2.drawContours(frame, [points], 0, (255, 0, 0), 2)

            # If the click counter hasn't reached 4, draw a circle on the cursor position.
            # Additionally, the if statement makes sure that it only draws the circle if the coords aren't 0, 0
            # because numpy arrays can't be empty, they're 0 by default, 0, 0 is the top left most corner of the screen.
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
                    raise NoSourceFound # If VideoCapture is corrupted or invalid, throws AttrErr
                

            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.rectangle(mask, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 0), thickness=cv2.FILLED)

            for spot_indx, spot in enumerate(self.spots):
                spot_status = self.spots_status[spot_indx]

                x1, y1, w, h = self.spots[spot_indx]
                points = np.array([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]])

                if spot_status:
                    frame = cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)
                else:
                    frame = cv2.drawContours(frame, [points], 0, (0, 0, 255), 2)

                cv2.rectangle(mask, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), thickness=cv2.FILLED)
            
            height, width, _ = frame.shape # Get screen size for font calculation

            font_scale = min(width, height) * FONT_SCALE
            thickness = int(np.ceil(min(width, height) * THICKNESS_SCALE))

            cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(self.spots_status)), str(len(self.spots_status))), (18, 18),
                cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), thickness=thickness, fontScale=font_scale) # Text Shadow
            cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(self.spots_status)), str(len(self.spots_status))), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=thickness, fontScale=font_scale) # Actual Text

            cv2.namedWindow('Parking', cv2.WINDOW_NORMAL)
            cv2.imshow('Parking', frame)
            cv2.setMouseCallback('Parking', self.mousePoints) # Required to make mouseclicks work.
            
            keypress = cv2.waitKey(25) & 0xFF

            if keypress == ord('q'):
                print("Exited with key 'q'")
                break

            if keypress == ord('s'):
                print("Saving mask...")
                self.save_mask(mask)

            # 13 is Enter
            if keypress == 13 and self.counter == 4 and self.new_spot is not None and previous_frame is not None:
                self.spots.append(self.new_spot)
                self.extend_slots(frame, previous_frame)

            # 8 is Backspace
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
    mask = None # Default mask to None instead of using the sample mask.

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
   

