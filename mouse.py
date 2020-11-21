from primitives import *


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
