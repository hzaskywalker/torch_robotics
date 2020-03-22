# shit...
import cv2
import numpy as np

class cv2Viewer(object):
    def __init__(self, width, height, display=None):
        self.width = width
        self.height = height

        self._geoms = []
        self.bbox = -1, 1, -1, 1 # l, r, b, t
        self.get_affine_transform()

    def get_affine_transform(self):
        l, r, b, t = self.bbox
        self.T = cv2.getAffineTransform(
            np.float32([[l, b], [l, t], [r, b]]),
            np.float32([[0, self.height], [0, 0], [self.width, self.height]])
        )

    def set_bounds(self, l, r, b, t):
        assert r-l == t-b
        self.bbox = l, r, b, t
        self.get_affine_transform()

    def add_onetime(self, geom):
        self._geoms.append(geom)

    def draw_circle(self, radius=10):
        geom = Circle(radius)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end):
        line = Line(start, end)
        self.add_onetime(line)
        return line

    def draw_polygon(self, v, filled=True):
        geom = Polygon(v=v, filled=filled)
        self.add_onetime(geom)
        return geom

    def Transform(self, rotation=0, translation=(0, 0)):
        return np.array([
                [np.cos(rotation), -np.sin(rotation), translation[0]],
                [np.sin(rotation), np.cos(rotation), translation[1]],
        ])

    def render(self, return_rgb_array=False):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

        for i in self._geoms:
            i.add_attr(self.T)
            i.render(img)
        self._geoms = []
        if return_rgb_array:
            return img
        else:
            cv2.imshow('x', img)
            cv2.waitKey(1)


def apply_transform(transform, p):
    return np.dot(transform, np.append(p, 1))


class Circle:
    def __init__(self, radius):
        self.radius = radius
        self.center = np.array([0, 0])
        self.color = (1, 1, 1)

    def add_attr(self, transform):
        # rotate first and then translate ..
        self.center = apply_transform(transform, self.center)
        self.radius *= np.abs(np.linalg.det(transform[:2,:2])) ** 0.5

    def set_color(self, *args):
        self.color = (np.array(args) * 255).astype(np.uint8)

    def render(self, img):
        color = tuple((int(i) for i in self.color))
        cv2.circle(img, (int(self.center[0]), int(self.center[1])), int(self.radius), color, -1)


class Polygon:
    def __init__(self, v, filled=True):
        self.points = np.array(v) # k x 2
        self.color = (1, 1, 1)
        self.filled = filled

    def add_attr(self, transform):
        # rotate first and then translate ..
        for i in range(len(self.points)):
            self.points[i] = apply_transform(transform, self.points[i])

    def set_color(self, *args):
        self.color = (np.array(args) * 255).astype(np.uint8)

    def render(self, img):
        p = [(int(i[0]), int(i[1])) for i in self.points]
        color = tuple((int(i) for i in self.color))
        cv2.fillPoly(img, np.int32([p]), color)


class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def add_attr(self, transform):
        # rotate first and then translate ..
        self.start = apply_transform(transform, self.start)
        self.end = apply_transform(transform, self.end)

    def render(self, img):
        p = [(int(i[0]), int(i[1])) for i in [self.start, self.end]]
        cv2.line(img, p[0], p[1], (0, 0, 0), 4)
