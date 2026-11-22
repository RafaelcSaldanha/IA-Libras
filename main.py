import cv2

from hand_tracking import HandDetector, HandVisualizer


def main():

    camera = cv2.VideoCapture(0)

    detector = HandDetector(num_hands=2, confidence=0.5)
    visualizer = HandVisualizer()

    if not camera.isOpened():
        print("Erro ao acessar a câmera.")
        detector.close()
        return

    print("Hand Tracking iniciado! Pressione Q para sair.")

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("Erro ao capturar imagem.")
                break

            frame = cv2.flip(frame, 1)

            results = detector.detect(frame)

            frame = visualizer.draw(
                frame,
                results,
                detector.connections
            )

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()