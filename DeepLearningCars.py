import math
import os
import timeit
from random import randint

import numpy as np
import pygame
from pygame.locals import *
from scipy.spatial import ConvexHull

# import dis

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20, 30)
pygame.init()

reset_timer = 30000  # reset the track after n ms
num_car = 200  # how many car to spawn
SCALED = 1


# The math, game engine, neural net code, and genetic algorithms are all encapsulated here!
# I know one continuous file with no documentation is awesome, so have fun exploring!

# But Seriously if you are interested in how it all works start at the bottom and work your way up,
# because if you think about it code is like a pyramid and the base of it is here at the top
# or just contact me (mason.hawver@gmail.com) also the game engine is based upon my c++ basic game engine to an extent


class Vector:
    """a basic vector class"""

    def __init__(self, x=0, y=0) -> None:
        self.x = x
        self.y = y

    def __iter__(self):
        return self

    def __getitem__(self, key: int) -> int or float:
        if key % 2 == 0:
            return self.x
        return self.y

    def __len__(self) -> int:
        """Always return 2"""
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
        return math.sqrt(pow(self.x, 2) + pow(self.y, 2))

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

    def getProjection(self, vec: 'Vector'):
        """ projects vec onto self """
        p = self.dot(vec) / (self.getMag() ** 2)
        proj = Vector(self.x, self.y)
        proj.mult((p, p))
        return proj

    def getReciprocal(self):
        """ rot 90 """
        return Vector(self.y, -1 * self.x)

    def rotate(self, a, c=(0, 0), degree=0):
        self.sub(c)
        a = a / (180.0 / math.pi) if degree else a
        x = (self.x * math.cos(a)) - (self.y * math.sin(a))
        y = (self.x * math.sin(a)) + (self.y * math.cos(a))
        self.set((x, y))
        self.add(c)

    def getAngleDiff(self, vec):
        val = self.dot(vec) / (self.getMag() * vec.getMag())
        val = max(min(val, 1), -1)
        angle = math.acos(val)  # in radians
        return angle

    def scale(self, scale, c=(0, 0)):
        self.sub(c)
        self.mult(scale)
        self.add(c)

    def __repr__(self):
        return "Vec = [x: {}, y: {}]".format(self.x, self.y)


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

    def scale(self, s):
        self.start.mult(s)
        self.end.mult(s)

    def draw(self, display):
        pygame.draw.line(display, self.color,
                         [int(self.start.x), int(self.start.y)],
                         [int(self.end.x), int(self.end.y)], self.width * 2 + 1)


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


class Viewer:
    def __init__(self, world, display=pygame.display.set_mode((800, 400), SCALED), size=(20, 10), pos=(-5, -5)):
        display.set_alpha(None)
        self.world = world
        self.display = display
        self.size = Vector()
        self.size.set(size)
        self.pos = Vector()
        self.pos.set(pos)

    def zoom(self, amt):
        old = Vector()
        old.set(self.size)
        self.size.mult((amt, amt))
        old.sub(self.size)
        old.div((2, 2))
        self.pos.add(old)

    def clear(self, color=(100, 180, 110)):
        self.display.fill(color)

    def draw(self, shapes):
        shift = Vector(self.pos.x, self.pos.y)
        scale = Vector()
        scale.set(self.display.get_size())
        scale.div(self.size)

        for shape in shapes:
            shape.shift(shift)
            shape.scale(scale)
            shape.draw(self.display)

        shift.mult((-1, -1))
        scale.flip()

        for shape in shapes:
            shape.scale(scale)
            shape.shift(shift)

    @staticmethod
    def render():
        pygame.display.flip()


class TargetViewer(Viewer):
    def __init__(self, world, target=(0, 0), display=pygame.display.set_mode((1000, 1000)), size=(800, 800),
                 pos=(-400, -400)):
        Viewer.__init__(self, world, display, size, pos)
        self.target = Vector()
        self.target.set(target)

        self.speed = 10

    def updatePos(self, target=None):
        if target is not None and not (pygame.mouse.get_pressed(3)[0]):
            self.target.mult((0, 0))
            self.target.sub(self.size)
            self.target.div((2, 2))
            self.target.add(target)

            vel = Vector()
            vel.set(self.target)
            vel.sub(self.pos)

            # vel.limitMag(self.speed)
            vel.mult((self.world.dt, self.world.dt))
            vel.mult((self.speed, self.speed))

            self.pos.add(vel)


class Mouse:
    def __init__(self, world):
        self.world = world
        self.body = Circle(10)

        self.delta = Vector()

        pygame.mouse.set_visible(True)

    @staticmethod
    def getScreenPos():
        pos = Vector()
        pos.set(pygame.mouse.get_pos())
        return pos

    def updatePos(self):
        old = self.body.pos
        pos = Mouse.getScreenPos()
        pos.div(self.world.viewer.display.get_size())
        pos.mult(self.world.viewer.size)
        pos.add(self.world.viewer.pos)
        self.body.pos = pos
        self.delta.set(pos)
        self.delta.sub(old)

    def update(self):
        self.updatePos()

        if pygame.mouse.get_pressed(3)[0]:
            self.world.viewer.pos.sub(self.delta)
            self.body.pos.sub(self.delta)

        for event in self.world.events:
            if event.type == 5:
                amt = .1

                if event.button == 4:
                    self.world.viewer.zoom(1 + amt)
                elif event.button == 5:
                    self.world.viewer.zoom(1 - amt)

    def draw(self):
        # pygame.draw.circle(self.world.viewer.display, (0, 0, 0), Mouse.getScreenPos(), self.body.radius)
        pass


class PhysicsEntity:
    def __init__(self, world, pos=(0, 0), vel=(0, 0), acc=(0, 0)):
        self.world = world
        self.pos = Vector()
        self.pos.set(pos)
        self.vel = Vector()
        self.vel.set(vel)
        self.acc = Vector()
        self.acc.set(acc)

        self.mass = 150

    def applyForce(self, f=(0, 0)):
        self.acc.add((f[0] / self.mass, f[1] / self.mass))

    def applyFriction(self, f=.1):
        f = 1 - f
        f = pow(f, self.world.dt)
        self.vel.mult((f, f))

    def applyVel(self):
        v = Vector()
        v.set(self.vel)
        v.mult((self.world.dt, self.world.dt))

        self.pos.add(v)

    def applyAcc(self):
        a = Vector()
        a.set(self.acc)
        a.mult((self.world.dt, self.world.dt))

        self.vel.add(a)
        self.acc.mult((0, 0))

    def update(self):
        self.applyAcc()
        self.applyVel()


class Car(PhysicsEntity):
    def __init__(self, world):
        PhysicsEntity.__init__(self, world)
        self.body = Poly()
        self.body.setByRect(Rect())

        self.steering_angle = 0
        self.max_steering_angle = math.pi / 6

        # self.c_drag = .4257  # drag coef
        # self.c_rr = 12.8  # rolling resistance coef
        # self.c_tf = 12  # tyre friction coef
        self.c_drag = .4257  # drag coef
        self.c_rr = 12.8  # rolling resistance coef
        self.c_tf = 8  # tyre friction coef

        self.breakingForce = 200000  # KERU original is 100000
        self.engineForce = 700000  # KERU : original is 350000

        self.accelerating = 0
        self.breaking = 0
        self.turning = 0

    def updatePos(self):
        self.body.pos = self.pos

    def updateFriction(self):
        """ apply drag, rolling resistance, and lateral friction"""
        if abs(self.vel.x) > .01 and abs(self.vel.y) > .01:
            """ apply the drag force """
            drag = Vector()
            drag.set(self.vel)
            d = self.c_drag * self.vel.getMag() * -1
            drag.mult((d, d))

            self.applyForce(drag)

            """ apply the rolling resistance force """
            rr = Vector()
            rr.set(self.vel)
            rr.mult((self.c_rr, self.c_rr))
            rr.mult((-1, -1))

            self.applyForce(rr)

        """ apply lateral friction"""
        # ms = 250
        # self.c_tf = max(3, (((15 - 5) / (100 - ms)) * (self.vel.getMag() - ms)) + 3)

        line = -1 * self.c_tf * self.mass
        lateral = self.body.getMidPoint().getReciprocal()
        lat_vel = lateral.getProjection(self.vel)
        lat_vel.mult((line, line))

        self.applyForce(lat_vel)

    def applyBreakingForce(self, amt=1):
        """ apply the breaking force """
        if not (self.vel.x == 0 and self.vel.y == 0) and (not amt == 0):
            br = self.body.getMidPoint()
            br.setMag(self.breakingForce * -amt)

            self.applyForce(br)

    def applyEngineForce(self, amt=1):
        """ apply the traction force """
        forward = self.body.getMidPoint()
        forward.setMag(self.engineForce * amt)

        self.applyForce(forward)

    def rotate(self):
        front = self.body.getMidPoint()
        end = Vector()
        end.set(front)
        end.mult((-1, -1))

        heading = Vector()
        heading.add(front)
        heading.sub(end)

        heading2 = Vector()
        heading2.set(heading)
        v = Vector()
        v.set(self.vel)
        v.mult((self.world.dt, self.world.dt))
        heading2.sub(v)
        v.rotate(self.steering_angle)
        heading2.add(v)
        d = -1 if heading2.y * heading.x < heading.y * heading2.x else 1
        self.body.rotate(heading.getAngleDiff(heading2) * d)

        heading2.setMag(self.vel.getMag())
        # self.vel = heading2

    def turn(self, direction):
        new_sa = direction * self.max_steering_angle
        diff = new_sa - self.steering_angle
        self.steering_angle += (diff * self.world.dt * 10)
        self.rotate()

    def input(self):
        for event in self.world.events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.accelerating = 1

                if event.key == pygame.K_DOWN:
                    self.breaking = 1

                if event.key == pygame.K_RIGHT:
                    self.turning = 1

                if event.key == pygame.K_LEFT:
                    self.turning = -1

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.accelerating = 0

                if event.key == pygame.K_DOWN:
                    self.breaking = 0

                if event.key == pygame.K_RIGHT:
                    self.turning = 0

                if event.key == pygame.K_LEFT:
                    self.turning = 0

        if self.accelerating:
            self.applyEngineForce()

        if self.breaking:
            self.applyBreakingForce()

        if abs(self.turning):
            self.steering_angle = self.max_steering_angle * self.turning  # * self.world.dt
            # self.steering_angle = min(abs(self.steering_angle), self.max_steering_angle) * self.turning
            self.rotate()
        else:
            self.steering_angle = 0

    def colliding(self, line):
        lines = [Line(self.body.points[0], self.body.points[3]),
                 Line(self.body.points[1], self.body.points[2]),
                 Line(self.body.points[0], self.body.points[1])]

        for _line in lines:
            _line.shift((self.body.pos.x * -1, self.body.pos.y * -1))
            if _line.getLineIntercept(line) is not None:
                return 1

        return 0

    def update(self):
        self.updatePos()
        self.input()
        self.updateFriction()
        self.applyAcc()
        self.applyVel()

    def draw(self):

        end = self.body.getMidPoint()
        end.add(self.body.pos)
        start = Vector()
        start.set(self.body.pos)
        line = Line(start, end)

        self.world.viewer.draw([self.body, line])


class RaceCar(Car):
    def __init__(self, world):
        Car.__init__(self, world)

        self.current_track = 0

        self.last_gateTime = timeit.default_timer()

        self.stop = 0

    def start(self):
        ct = self.world.track.tracks[self.current_track]
        # ct.activate()
        self.pos.set(ct.getCenter())

        orientation = Vector()
        orientation.set(ct.next_track.gate.end)
        orientation.add(ct.next_track.gate.start)
        orientation.div((2, 2))
        self.world.viewer.draw([Circle(5, orientation)])
        orientation.sub(self.pos)

        for i in range(100):
            direction = self.body.getMidPoint()

            a = orientation.getAngleDiff(direction)
            a *= -1 if orientation.y / (direction.y * orientation.x) > direction.x else 1

            self.body.rotate(a)

            line_direction = self.body.getMidPoint()

            if line_direction.x == direction.x and line_direction.y == direction.y:
                break

    def getCurrentGate(self):
        return self.world.track.tracks[self.current_track]

    def updateCurrentTrack(self):
        ct = self.world.track.tracks[self.current_track]
        # ct.activate()

        if self.colliding(ct.next_track.gate):
            self.nextGate()

    def nextGate(self):
        self.current_track = (self.current_track + 1) % len(self.world.track.tracks)

    def updateCol(self):
        ct = self.world.track.tracks[self.current_track]
        lt = self.world.track.tracks[(self.current_track - 1) % len(self.world.track.tracks)]

        for line in [ct.track_inner, ct.track_outer, lt.track_inner, lt.track_outer, lt.gate]:
            if self.colliding(line):
                self.crash()

    def crash(self):
        self.stop = 1

    def update(self):
        if not self.stop:
            # self.start()
            self.updatePos()
            self.input()
            self.updateFriction()
            self.applyAcc()
            self.applyVel()
            self.updateCurrentTrack()
            self.updateCol()


def relu(val):
    return min(max(0, val), 100)


def tanh(val):
    return (2.0 / (1 + math.pow(math.e, -2 * val / 10.0))) - 1


class Dense:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros(output_size)

        self.output = []

        self.activation = activation

    def call(self, _input):
        output = np.array([_input[i] * self.weights[i] for i in range(len(self.weights))])
        output += self.bias

        output = np.array([np.sum(output[:, i]) for i in range(len(self.weights[0]))])
        output = np.array([self.activation(val) for val in output])

        self.output = output

        return output

    def setRandomWeights(self, amt):
        self.weights = np.random.normal(0, amt, self.weights.shape)

    def setRandomBiases(self, amt):
        self.bias = np.random.normal(0, amt, self.bias.shape)


class AutoBrain:
    def __init__(self, auto):
        self.auto = auto
        self.num_input = 8
        self.input = []
        self.dense1 = Dense(self.num_input, 12, relu)  # KERU speed
        self.dense2 = Dense(12, 6, relu)
        self.out = Dense(6, 2, tanh)

        self.randomize(self.auto.world.learning_rate)  # KERU

    def call(self, _input):
        self.input = _input
        _input = np.array(_input)
        output = self.dense1.call(_input)
        output = self.dense2.call(output)
        output = self.out.call(output)

        return output

    def randomize(self, amt):
        self.dense1.setRandomWeights(amt)
        self.dense2.setRandomWeights(amt)
        self.out.setRandomWeights(amt)

        self.dense1.setRandomBiases(amt)
        self.dense2.setRandomBiases(amt)
        self.out.setRandomBiases(amt)

    def mutate(self, parent, amt):
        self.randomize(amt)

        self.dense1.weights += parent.dense1.weights
        self.dense1.bias += parent.dense1.bias
        self.dense2.weights += parent.dense2.weights
        self.dense2.bias += parent.dense2.bias
        self.out.weights += parent.out.weights
        self.out.bias += parent.out.bias

    def draw(self, bounding_rect):
        layers = [self.dense1, self.dense2, self.out]
        width = bounding_rect.size.x / (len(layers) + 1)

        last_layer = []

        height = bounding_rect.size.y / len(self.input)
        pos = Vector()
        pos.set(bounding_rect.pos)

        # size = self.num_input #KERU speed

        for _input in self.input:
            activation = max(min((_input / 100), 1), 0)
            color = int(activation * 255)

            unit = Circle(6, pos, (color, color, color))
            unit.pos.add((0, height / 2.0))
            self.auto.world.viewer.draw([unit])
            pos.add((0, height))

            last_layer.append(unit)

        for i, layer in enumerate(layers):
            if len(layer.output) > 0:
                height = bounding_rect.size.y / len(layer.output)
                pos = Vector()
                pos.set(bounding_rect.pos)
                pos.add((width * (i + 1), 0))

                units = []
                conns = []

                for j, val in enumerate(layer.output):
                    if i < 2:
                        activation = max(min((val / 100.0), 1), 0)
                    else:
                        activation = max(min(((val + 1) / 2.0), 1), 0)

                    color = int(activation * 255)

                    unit = Circle(5, pos, (color, color, color))
                    unit.pos.add((0, height / 2.0))
                    # self.auto.world.viewer.draw([unit])
                    pos.add((0, height))

                    for k, lunit in enumerate(last_layer):
                        activation = max(min(((abs(layer.weights[k][j])) / .2), 1), 0)
                        w = int(activation)  # KERU was 3 * activation + 1
                        # conn = Line(lunit.pos, unit.pos, (0, 0, 0), w)
                        conn = Line(lunit.pos, unit.pos, (activation * 128, activation, activation), w)
                        conns.append(conn)
                        # self.auto.world.viewer.draw([conn])

                    units.append(unit)

                self.auto.world.viewer.draw(conns)
                self.auto.world.viewer.draw(last_layer)

                last_layer = units

        self.auto.world.viewer.draw(last_layer)


class Auto(RaceCar):
    next_id = 0

    def __init__(self, world, parent=None):
        RaceCar.__init__(self, world)
        self.start_time = pygame.time.get_ticks()
        self.current_track = 0
        self.top = 0

        self.brain = AutoBrain(self)

        self.engine_amt = 0  # 0 - 1
        self.break_amt = 0  # 0 - 1
        self.turn_amt = 0  # -1 - 1

        self.fitness = 0

        self.inputs = []

        self.gen = world.generation

        self.parent = parent

        self.id = Auto.next_id
        Auto.next_id += 1

    def getName(self):
        return "Auto_{}".format(self.id) if self.parent is None else "{}_{}".format(self.parent.name, self.id)

    def start(self):
        self.fitness = 0
        self.current_track = 0
        self.stop = 0
        self.vel.set((0, 0))
        self.acc.set((0, 0))
        RaceCar.start(self)
        self.getCurrentGate().activate()
        self.stop = 0
        self.start_time = pygame.time.get_ticks()

    def nextGate(self):
        if self.top:
            self.getCurrentGate().deactivate()
        RaceCar.nextGate(self)
        if self.top:
            self.getCurrentGate().activate()
        self.fitness += 1

    def getInputs(self):
        max_dis = 2000

        angles = [-90, -40, -20, 0, 20, 40, 90]
        self.inputs = []
        inputs = []

        for angle in angles:
            start = Vector()
            start.set(self.pos)
            end = self.body.getMidPoint(0, 1)
            end.rotate(angle, degree=True)
            end.setMag(max_dis)
            end.add(start)

            ray = Line(start, end)
            col = None
            i = 0

            cols = []  # KERU collision fix ?
            while i < len(self.world.track.tracks) and col is None:
                index = math.ceil(i / 2.0) * (((i % 2) * 2) - 1)
                track = self.world.track.tracks[(index + self.current_track) % len(self.world.track.tracks)]
                walls = [track.track_inner, track.track_outer]

                cols = []  # should it be here ?

                for wall in walls:
                    c = ray.getLineIntercept(wall)
                    if c is not None:
                        cols.append(c)

                min_dis = None
                col = None
                for c in cols:
                    dis = start.getDis(c)
                    if min_dis is None or dis < min_dis:
                        min_dis = dis
                        col = c

                i += 1
            # print(cols)
            if pygame.time.get_ticks() - self.start_time > 2000:
                if not cols:  # same as if len(cols) == 0
                    self.stop = 1  # KERU collision fix

            if col is None:
                col = end

            if self.top or 0:
                self.inputs.append(Circle(5, col))
                # self.inputs.append(Line(start, col, (255, 255, 255)))

            dis = start.getDis(col)
            inputs.append(dis)

        return inputs

    def getOutput(self, _input):
        output = self.brain.call(_input)

        self.engine_amt = max(0, output[0])
        self.break_amt = min(0, output[0])
        self.turn_amt = output[1]

    def act(self):
        _input = self.getInputs()
        # if self.vel.getMag() != 0: print(self.vel.getMag() / 255) # KERU speed
        _input.append(self.vel.getMag() / 20)  # KERU speed
        self.getOutput(_input)

        self.applyBreakingForce(self.break_amt)
        # print(self.world.dt)
        self.applyEngineForce(self.engine_amt)
        self.turn(self.turn_amt)

    def update(self):
        if not self.stop:
            self.act()
            self.updateFriction()
            self.applyAcc()
            self.applyVel()
            self.updatePos()
            self.updateCurrentTrack()
            self.updateCol()

        c = (51, 51, 51)
        if self.gen != self.world.generation - 1:
            c = (200, 180, 140)
        elif self.top:
            c = (20, 255, 230)

        self.body.color = c

    def makeChild(self):
        child = Auto(self.world, self)
        child.brain.mutate(self.brain, self.world.learning_rate)

        return child


class Track:
    def __init__(self, so=(0, 0), si=(0, 0), eo=(0, 0), ei=(0, 0), nt=None):
        self.track_outer = Line(so, eo)
        self.track_inner = Line(si, ei)
        self.gate = Line(so, si)

        self.next_track = nt

        self.color = (0, 0, 0)

    def activate(self):
        self.color = (0, 0, 255)
        self.next_track.color = (0, 255, 0)

    def deactivate(self):
        self.color = (0, 0, 0)
        self.next_track.color = (0, 0, 0)

    def getCenter(self):
        center = Vector()
        for point in [self.track_outer.start, self.track_outer.end, self.track_inner.start, self.track_inner.end]:
            center.add(point)

        center.div((4, 4))
        return center


class RaceTrack:
    def __init__(self, world):
        self.tracks = []
        self.world = world

        self.dir = (randint(0, 1) * 2) - 1

    def generateTrack(self, bounding_rect, width):
        self.tracks = []

        track_outer = Poly()
        track_inner = Poly()

        track_outer.points = []
        track_inner.points = []

        # points = np.random.rand(randint(20, 30), 2)
        points = np.random.rand(randint(20, 30), 2)  # KERU
        # print(points)
        hull = ConvexHull(points)

        self.dir = (randint(0, 1) * 2) - 1

        verts = hull.vertices  # [::self.dir]

        for vert in verts:
            point = points[vert]
            point -= [.5, .5]
            point = (point * [bounding_rect.size.x, bounding_rect.size.y])
            point = Vector(point[0], point[1])
            d = 100
            for p in track_outer.points:
                d = Vector()
                d.set(point)
                d.sub(p)
                d = d.getMag()
                if d < 80:
                    break
            if d > 80:
                track_outer.points.append(point)
                track_inner.points.append(Vector(point[0], point[1]))

        track_outer.pos = bounding_rect.getCenter()
        track_inner.pos = bounding_rect.getCenter()

        # boundingRect = self.track_outer.getBoundingRect()

        # scale = Vector(width, width)
        # scale.sub(boundingRect.size)
        # scale.div(boundingRect.size)
        # scale.mult((-1, -1))

        scale = -1 * ((width / min(bounding_rect.size.x, bounding_rect.size.y)) - 1)

        track_inner.shift(bounding_rect.pos)
        track_inner.scale((scale, scale))

        shift = Vector()
        shift.set(bounding_rect.size)
        shift.mult((1 - scale, 1 - scale))
        shift.div((-2, -2))
        track_inner.shift(shift)
        track_inner.shift((bounding_rect.pos.x * -1, bounding_rect.pos.y * -1))

        for i in range(len(track_outer.points)):
            si = Vector()
            si.set(track_inner.points[i - 1])
            so = Vector()
            so.set(track_outer.points[i - 1])
            ei = Vector()
            ei.set(track_inner.points[i])
            eo = Vector()
            eo.set(track_outer.points[i])
            si.add(track_inner.pos)
            ei.add(track_inner.pos)
            so.add(track_outer.pos)
            eo.add(track_outer.pos)
            track = Track(so, si, eo, ei)
            self.tracks.append(track)

            if len(self.tracks) > 1:
                self.tracks[len(self.tracks) - 2].next_track = track

        self.tracks[len(self.tracks) - 1].next_track = self.tracks[0]

    def draw(self):
        circles = []
        lines = []  # [Line(self.points[0], self.points[-1])]

        for track in self.tracks:
            lines.append(track.track_outer)
            lines.append(track.track_inner)
            lines.append(track.gate)

            circles.append(Circle(5, track.track_inner.start, track.color))
            circles.append(Circle(5, track.track_outer.start, track.color))

        self.world.viewer.draw(circles + lines)


class AutoSimulation:
    def __init__(self):
        self.viewer = TargetViewer(self)
        self.mouse = Mouse(self)
        self.text = Text("Generation : 0")
        self.autos = []
        self.top_auto = None
        self.track = RaceTrack(self)

        self.learning_rate = .1  # KERU original was .1

        self.generation = 0

        self.events = []
        self.dt = 0

        self.start_time = pygame.time.get_ticks()

        self.addNewAutos(num_car)

    def generateTrack(self, difficulty):
        size = randint(int(800 - (400 * difficulty)), 800)
        width = size / 4  # KERU was / 4
        self.track.generateTrack(Rect((size, size), (-size / 2.0, -size / 2.0)), width)

    def addNewAutos(self, n):
        for i in range(n):
            self.autos.append(Auto(self))

    #    @staticmethod
    #    def loadAutos(files=[]):
    #        pass

    def getMostFitAuto(self):
        # print("----")
        max_fit = -1
        top_autos = []
        for auto in self.autos:
            auto.top = 0
            if auto.fitness > max_fit:
                max_fit = auto.fitness
                top_autos = [auto]

            if not auto.stop:
                pass
                # print(auto.fitness)

            if auto.fitness == max_fit:
                top_autos.append(auto)

        if len(top_autos) == 1:
            return top_autos[0]

        gate_pos = top_autos[0].getCurrentGate().next_track.gate.getCenter()

        closest = None
        top_auto = None

        for auto in top_autos:
            dis = gate_pos.getDis(auto.pos)
            if closest is None or dis < closest:
                closest = dis
                top_auto = auto

        top_auto.top = 1

        self.top_auto = top_auto

    def start(self):
        d = 1.0 - (1.0 / ((self.generation / 5.0) + 1.0))
        self.generateTrack(d)
        self.text = Text("Generation : " + str(self.generation))
        self.start_time = pygame.time.get_ticks()
        for car in self.autos:
            car.start()

    def update(self):
        for car in self.autos:
            car.update()
            if pygame.time.get_ticks() - self.start_time > 5000:  # KERU
                if car.vel.getMag() < 4:
                    # print(car.vel.getMag())
                    car.stop = 1
        self.getMostFitAuto()
        self.mouse.update()
        if not self.top_auto.stop:
            self.viewer.updatePos(self.top_auto.body.pos)
        # self.generateTrack()

        if pygame.time.get_ticks() - self.start_time > reset_timer:
            self.reset()

    def cont(self):
        stopped = 0
        for car in self.autos:
            stopped += car.stop

        all_stopped = stopped == (len(self.autos) - 1)  # KERU

        # print(len(self.autos), stopped)

        return 1 - all_stopped

    def end(self):

        new_batch = []

        # count car that are still running and make mutated child of them
        running_car_count = 0
        for running_auto in self.autos:
            if running_auto.stop == 0:
                running_car_count = running_car_count + 1
                # new_batch.append(running_auto.makeChild())
                new_batch.append(running_auto)

        print("running cars : ", running_car_count)

        # complete the list with mutated child of the top auto
        for i in range(len(self.autos) - running_car_count):
            auto = self.top_auto.makeChild()
            new_batch.append(auto)

        # finally, copy the new batch + the top auto (without mutating it) to the next batch
        self.autos = new_batch + [self.top_auto]

        self.generation += 1

        print("Generation:", self.generation - 1, "Done!")

    def render(self):
        for car in self.autos:
            if not car.stop:  # KERU
                car.draw()
        self.top_auto.draw()
        self.top_auto.brain.draw(Rect((200, 150), (self.viewer.pos.x + 50, self.viewer.pos.y + 50)))
        self.track.draw()
        self.mouse.draw()

        self.text.draw(self.viewer.display)

        self.top_auto.brain.draw(Rect((200, 150), (self.viewer.pos.x + 50, self.viewer.pos.y + 50)))
        # KERU self.viewer.draw(self.top_auto.inputs)
        self.viewer.render()
        self.viewer.clear()

    def reset(self):
        self.end()
        self.start()


class Game:
    def __init__(self, world=AutoSimulation()):
        self.world = world
        self.fps = 60
        self.running = 1

        self.t = timeit.default_timer()

    def update(self):
        self.world.events = pygame.event.get()
        self.world.dt = 0.01  # (timeit.default_timer() - self.t)
        self.world.update()
        self.t = timeit.default_timer()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.world.reset()

    def render(self):
        self.world.render()

    def play(self):

        running = True
        lr = timeit.default_timer()

        self.world.start()

        while running and self.running:
            self.update()
            if self.t - lr > 1. / self.fps:
                self.render()

                lr = timeit.default_timer()

            running = self.world.cont()

        self.world.end()

        return


def main():
    g = Game()
    g.world.generation += 1
    while g.running:
        g.play()


# dis.dis(Line.getSlope)
main()
