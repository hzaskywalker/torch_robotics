from robot import renderer
import cv2

def test_sphere():
    r = renderer.Renderer()

    r.add_sphere((0, 0, 0), 3, (255, 255, 255))
    r.add_point_light((0, 5, 0), color=(255, 0, 0))

    r.set_camera_position(0, 0, 10)

    img = r.render()

    cv2.imwrite('x.jpg', img)
    #while True:
    #    r.render(mode='human')

if __name__:
    test_sphere()