import math


class Vector:
    """a basic vector class"""

    def __init__(self, x=0, y=0) -> None:
        self.x = x
        self.y = y

    def __iter__(self):
        return self

    def __getitem__(self, key: int) -> int or float:
        return self.x if key == 0 else self.y

    def __len__(self) -> int:
        return 2

    def set(self, iterable) -> None:
        self.x = iterable[0]
        self.y = iterable[1]

    def add(self, vec) -> None:
        self.x += vec[0]
        self.y += vec[1]

    def sub(self, vec) -> None:
        self.x -= vec[0]
        self.y -= vec[1]

    def mult(self, vec) -> None:
        self.x *= vec[0]
        self.y *= vec[1]

    def div(self, vec) -> None:
        try:
            self.x /= vec[0]
            self.y /= vec[1]
        except ZeroDivisionError:
            print("No div by 0")

    def flip(self) -> None:  # forgot the real name for this...
        self.x = 1. / self.x
        self.y = 1. / self.y

    def getDis(self, vec) -> int or float:
        dis = Vector()
        dis.set(self)
        dis.sub(vec)
        return dis.getMag()

    def normalize(self) -> None:
        self.setMag(1)

    def getMag(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def setMag(self, mag) -> None:
        mag = mag / self.getMag()
        self.mult(Vector(mag, mag))

    def limitMag(self, max_mag) -> None:
        mag = self.getMag()
        if mag > max_mag:
            mag *= max_mag
            self.div((mag, mag))

    def dot(self, vec) -> int or float:
        return (self.x * vec.x) + (vec.y * self.y)

    def getProjection(self, vec: 'Vector') -> 'Vector':
        """ projects vec onto self """
        p = self.dot(vec) / (self.getMag() ** 2)
        proj = Vector(self.x, self.y)
        proj.mult((p, p))
        return proj

    def getReciprocal(self) -> 'Vector':
        """ rot 90 """
        return Vector(self.y, -1 * self.x)

    def rotate(self, a, c=(0, 0), degree=0) -> None:
        self.sub(c)
        a = a / (180.0 / math.pi) if degree else a
        x = (self.x * math.cos(a)) - (self.y * math.sin(a))
        y = (self.x * math.sin(a)) + (self.y * math.cos(a))
        self.set((x, y))
        self.add(c)

    def getAngleDiff(self, vec) -> int or float:
        val = self.dot(vec) / (self.getMag() * vec.getMag())
        val = max(min(val, 1), -1)
        angle = math.acos(val)  # in radians
        return angle

    def scale(self, scale, c=(0, 0)) -> None:
        self.sub(c)
        self.mult(scale)
        self.add(c)

    def __repr__(self) -> str:
        return "Vec = [x: {}, y: {}]".format(self.x, self.y)
