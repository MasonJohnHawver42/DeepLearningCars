import pygame

from vector import Vector


class Line:
    def __init__(self, s=(0, 0), e=(1, 1), color=(255, 0, 0), width=1):
        self.start = Vector()
        self.start.set(s)
        self.end = Vector()
        self.end.set(e)

        self.width = width
        self.color = color

    def getSlope(self):
        try:
            return (self.start.y - self.end.y) / (self.start.x - self.end.x)
        except ZeroDivisionError:
            return None

    def getIntercept(self):
        # m = self.getSlope()
        # if m is not None:
        #    return self.start.y - (m * self.start.x)
        # return None
        try:
            return self.start.y - (((self.start.y - self.end.y) / (self.start.x - self.end.x)) * self.start.x)
        except ZeroDivisionError:
            return None

    def inDomain(self, x):
        return min(self.start.x, self.end.x) < x < max(self.start.x, self.end.x)

    # TODO: optimize this ! (14% CPU time)
    def getLineIntercept(self, line):
        m1 = self.getSlope()
        b1 = self.getIntercept()
        m2 = line.getSlope()
        b2 = line.getIntercept()

        try:
            xi = (b2 - b1) / (m1 - m2)
        except (ZeroDivisionError, TypeError) as e:
            print("Exception in GetLineIntercept : %s", e)
            return None

        if self.inDomain(xi) and line.inDomain(xi):
            yi = (m1 * xi) + b1
            return Vector(xi, yi)
        else:
            return None

    def getCenter(self):
        c = Vector()
        c.set(self.start)
        c.add(self.end)
        c.div((2, 2))

        return c

    def shift(self, s):
        self.start.sub(s)
        self.end.sub(s)

    # TODO : slow ! (only 3% but... meh)
    def scale(self, s):
        self.start.mult(s)
        self.end.mult(s)

    def draw(self, display):
        pygame.draw.line(display, self.color,
                         [int(self.start.x), int(self.start.y)],
                         [int(self.end.x), int(self.end.y)], self.width * 2 + 1)


class Text:
    def __init__(self, message):
        self.font = pygame.font.Font('barcade-brawl.ttf', 16)
        self.text = self.font.render(message, False, (0, 0, 0))
        self.textRect = self.text.get_rect()
        self.textRect.center = (955 - (self.textRect.size[0] / 2.0), 100 - (self.textRect.size[1] / 2.0))

    def scale(self, s):
        pass

    def shift(self, s):
        pass

    def draw(self, display):
        display.blit(self.text, self.textRect)


class Circle:
    def __init__(self, radius=1, pos=(0, 0), c=(0, 0, 0)):
        self.radius = radius
        self.pos = Vector()
        self.pos.set(pos)

        self.color = c

    def shift(self, s):
        self.pos.sub(s)

    def scale(self, s):
        self.pos.mult(s)
        self.radius *= min(s.x, s.y)

    def draw(self, display):
        pygame.draw.circle(display, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))


class Poly:
    def __init__(self, points=None, c=(0, 0)):
        if points is None:
            points = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

        self.points = []
        for point in points:
            p = Vector()
            p.set(point)
            self.points.append(p)
        self.pos = Vector()
        self.pos.set(c)

        self.color = (0, 0, 0)

    def setByRect(self, rect):
        pattern = [[-.5, .5],
                   [.5, .5],
                   [.5, -.5],
                   [-.5, -.5]]

        self.points = []

        for mask in pattern:
            point = Vector()
            point.set(mask)
            point.mult(rect.size)

            self.points.append(point)

        self.pos = rect.pos

    def getBoundingRect(self):
        top = None
        bottom = None
        right = None
        left = None

        for point in self.points:
            p = Vector()
            p.set(point)
            p.add(self.pos)

            if top is None or p.y > top:
                top = p.y

            if bottom is None or p.y < bottom:
                bottom = p.y

            if right is None or p.x > right:
                right = p.x

            if left is None or p.x < left:
                left = p.x

        return Rect(size=(right - left, top - bottom), pos=(left, top))

    def getMidPoint(self, i1=0, i2=1):
        mid = Vector()
        mid.set(self.points[i1])
        mid.add(self.points[i2])
        mid.div((2, 2))

        return mid

    def rotate(self, a):
        for point in self.points:
            point.rotate(a)

    def shift(self, shift):
        self.pos.sub(shift)

    def scale(self, scale):
        for point in self.points:
            point.add(self.pos)
            point.mult(scale)
            point.sub(self.pos)

    def draw(self, display):
        points = []
        for p in self.points:
            points.append([p.x + self.pos.x, p.y + self.pos.y])

        pygame.draw.polygon(display, self.color, points)


class Rect:
    def __init__(self, size=(7.5, 15), pos=(0, 0)):
        self.size = Vector()
        self.size.set(size)
        self.pos = Vector()
        self.pos.set(pos)

    def getCenter(self):
        c = Vector(0, 0)
        c.add(self.size)
        c.div(Vector(2, 2))
        c.add(self.pos)
        return c

    def shift(self, shift):
        self.pos.sub(shift)

    def scale(self, scale):
        self.size.mult(scale)
