import os
import pygame

from pygame.locals import *
from vector import Vector

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20, 30)


class Viewer:
    def __init__(self, world, display=pygame.display.set_mode((800, 400), SCALED), size=(20, 10), pos=(-5, -5)):
        display.set_alpha(None)
        self.world = world
        self.display = display
        self.size = Vector()
        self.size.set(size)
        self.pos = Vector()
        self.pos.set(pos)
        self.display_size = self.display.get_size()

    def zoom(self, amt):
        old = Vector()
        old.set(self.size)
        self.size.mult((amt, amt))
        old.sub(self.size)
        old.div((2, 2))
        self.pos.add(old)

    def clear(self, color=(100, 180, 110)):
        self.display.fill(color)

    # TODO : optimize this ! (14% CPU time)
    def draw(self, shapes):
        shift = Vector(self.pos.x, self.pos.y)
        scale = Vector()
        scale.set(self.display_size)
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
    def __init__(self, world, target=(0, 0), display=pygame.display.set_mode((1280, 720), RESIZABLE | DOUBLEBUF),
                 size=(1280, 720), pos=(-400, -400)):
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
