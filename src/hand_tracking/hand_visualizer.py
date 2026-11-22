import cv2


class HandVisualizer:

    def __init__(
        self,
        landmark_color=(0, 255, 0),
        connection_color=(255, 0, 0),
        landmark_radius=5,
        connection_thickness=2
    ):
        self.landmark_color = landmark_color
        self.connection_color = connection_color
        self.landmark_radius = landmark_radius
        self.connection_thickness = connection_thickness

    def draw(self, frame, results, connections):

        height, width, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:

            # Desenha os pontos
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    self.landmark_radius,
                    self.landmark_color,
                    -1
                )

            # Desenha as conexões
            for connection in connections:

                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                start_point = (
                    int(start.x * width),
                    int(start.y * height)
                )

                end_point = (
                    int(end.x * width),
                    int(end.y * height)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    self.connection_color,
                    self.connection_thickness
                )

        return frame