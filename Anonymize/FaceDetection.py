import cv2
from utils.utils import imageBlur


class FaceDetection:
    @staticmethod
    def FaceDetection(img, trackData):
        detections = []
        for track in trackData:
            if track[2] == "person" or track[2] == "rider":
                det = track.copy()
                det[2] = "face"
                xmin, ymin, xmax, ymax = (3,4,5,6)
                width = track[xmax] - track[xmin]
                height = (track[ymax] - track[ymin]) / 4
                if width < 10 or height < 10:
                    continue

                det[xmin] = int(track[xmin] + width * 0.1)
                det[xmax] = int(track[xmin] + width * 0.9)
                det[ymin] = int(track[ymin] + height * 0.1)
                det[ymax] = int(track[ymin] + height * 0.9)

                face = img[det[ymin]:det[ymax], det[xmin]:det[xmax]]
                # face = imageBlur(face)
                img[det[ymin]:det[ymax], det[xmin]:det[xmax]] = face

                detections.append(det)
        return detections, img

