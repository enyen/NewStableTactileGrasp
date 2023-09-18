import cv2
import time
import signal
import warnings
import numpy as np
from threading import Thread
from matcher import Matching


class MarkerFlow:
    def __init__(self, fps=30):
        # init param
        self.fps = fps
        self.cam_idx = [-1, -1]
        self.running = False
        self.collection = None
        self.process = Thread()

        # init process
        signal.signal(signal.SIGINT, self._signal_stop)
        self.get_cam_idx()

    def get_cam_idx(self):
        self.cam_idx = [-1, -1]
        self._inspect_cam()  # get cam1 id
        self._inspect_cam()  # get cam2 id
        assert self.cam_idx[0] != -1 and self.cam_idx[1] != -1, (
            'Camera id {} undefined(-1)!'.format(self.cam_idx))

    def _inspect_cam(self):
        for i in range(8):
            if i in self.cam_idx:
                continue
            cam = cv2.VideoCapture(i)
            if not cam.isOpened():
                continue
            for j in range(3 * 30):
                _, frame = cam.read()
                cv2.imshow('Identify Image source(left/right) for 3 seconds...', frame)
                cv2.waitKey(1000 // 30)
            print(frame.shape)
            cam.release()
            cv2.destroyAllWindows()
            idx = input('Enter "0" for left sensor, or "1" fo right sensor: ')
            self.cam_idx[int(idx)] = i
            break

    def start(self, vis=False):
        if not self.process.is_alive():
            self.process = Thread(target=self._run, args=(vis,))
            self.process.start()
        else:
            warnings.warn("Process already started!")

    def stop(self):
        if self.process.is_alive():
            self.running = False
            self.process.join(5)
            if self.process.is_alive():
                raise Exception("Unable to terminate thread!")
        else:
            warnings.warn("No active process to stop!")

    def _run(self, debug):
        self.running = True
        flows = []

        # matcher
        matcherl = Matching(N_=8, M_=6, fps_=self.fps,
                            x0_=25, y0_=30,
                            dx_=38, dy_=38)
        matcherr = Matching(N_=8, M_=6, fps_=self.fps,
                            x0_=25, y0_=30,
                            dx_=38, dy_=38)

        # camera
        caml = cv2.VideoCapture(self.cam_idx[0])
        camr = cv2.VideoCapture(self.cam_idx[1])
        caml.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        camr.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        caml.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        camr.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        while self.running:
            # image
            tick = time.time()
            imgl = caml.read()[1]
            imgr = camr.read()[1]
            imgl = cv2.resize(imgl, (320, 240), interpolation=cv2.INTER_AREA).astype(np.uint8)
            imgr = cv2.resize(imgr, (320, 240), interpolation=cv2.INTER_AREA).astype(np.uint8)
            imgl = cv2.rotate(imgl, cv2.ROTATE_90_CLOCKWISE)
            imgr = cv2.rotate(imgr, cv2.ROTATE_90_CLOCKWISE)

            # marker flow
            ctrl = self._marker_center(imgl)
            ctrr = self._marker_center(imgr)
            matcherl.init(ctrl)
            matcherr.init(ctrr)
            matcherl.run()
            matcherr.run()
            flowl = matcherl.get_flow()
            flowr = matcherr.get_flow()
            flows.append(self._convert_flows(flowl, flowr))

            # debug view
            if debug:
                self._draw_flow(imgl, flowl)
                self._draw_flow(imgr, flowr)
                for ctrs in zip(ctrl, ctrr):
                    cv2.circle(imgl, (int(ctrs[0][0]), int(ctrs[0][1])), 10, (255, 255, 255), 2, 6)
                    cv2.circle(imgr, (int(ctrs[1][0]), int(ctrs[1][1])), 10, (255, 255, 255), 2, 6)
                cv2.imshow('flow_left', imgl)
                cv2.imshow('flow_right', imgr)
                cv2.waitKey(1)

            # wait camera frame-rate
            time.sleep(max(1. / 30 - (time.time() - tick), 0))  # 30hz

        caml.release()
        camr.release()
        cv2.destroyAllWindows()
        self.running = False
        self.collection = np.stack(flows, axis=0)  # [t s c h w]

    def get_marker_flow(self):
        return self.collection.copy()

    @staticmethod
    def _convert_flows(fl, fr):
        """
        output: (Ox, Oy, Cx, Cy, Occupied) = flow
        Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
        Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
        Occupied: N*M matrix, the index of the marker at each position, -1 means inferred.
            e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
        :return drow_dcol in shape [s c h w], s:sensor(l,r), c:channel(drow,dcol), h:height, w:width
        """
        fl = np.asarray(fl)
        fr = np.asarray(fr)
        flow = np.stack((np.stack((fl[2] - fl[0], fl[3] - fl[1]), axis=0),
                         np.stack((fr[2] - fr[0], fr[3] - fr[1]), axis=0)), axis=0)
        return flow

    @staticmethod
    def _marker_center(frame):
        area_l, area_h = 6, 64

        # mask
        # subtract the surrounding pixels to magnify difference between markers and background
        mask = cv2.GaussianBlur(frame, (31, 31), 0).astype(np.float32) - cv2.GaussianBlur(frame, (5, 5), 0)
        mask = np.clip(mask * 8, 0, 255).astype(np.uint8)
        mask = cv2.inRange(mask, (200, 200, 200), (255, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        # center
        ctrs = []
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours[0]) < 19:  # if too little markers, then give up
            print("Too less markers detected: ", len(contours))
            return ctrs

        for contour in contours[0]:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (area_l < area < area_h) and (abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1):
                t = cv2.moments(contour)
                centroid = [t['m10'] / t['m00'], t['m01'] / t['m00']]
                ctrs.append(centroid)
        return ctrs

    @staticmethod
    def _draw_flow(frame, flow):
        Ox, Oy, Cx, Cy, Occupied = flow
        for i in range(len(Ox)):
            for j in range(len(Ox[i])):
                pt1 = (int(Ox[i][j]), int(Oy[i][j]))
                pt2 = (int(Cx[i][j]), int(Cy[i][j]))
                color = (0, 0, 255)
                if Occupied[i][j] <= -1:
                    color = (127, 127, 255)
                cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)

    def _signal_stop(self, signum, frame):
        self.running = False


if __name__ == "__main__":
    mf = MarkerFlow()
    mf.start()
    print(mf.get_marker_flow().shape)
