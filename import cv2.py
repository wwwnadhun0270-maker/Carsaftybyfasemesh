import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode  
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, 
                                                  max_num_faces=self.maxFaces, 
                                                  min_detection_confidence=self.minDetectionCon, 
                                                  min_tracking_confidence=self.minTrackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                         self.drawSpecs, self.drawSpecs)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append([cx, cy])

                faces.append(face)    

        return img, faces   


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        
        if not success:
            break
        
        img, faces = detector.findFaceMesh(img, True)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        cv2.putText(img, f'FPS:{int(fps)}', (28, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.putText(img, f'Faces:{len(faces)}', (28, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()