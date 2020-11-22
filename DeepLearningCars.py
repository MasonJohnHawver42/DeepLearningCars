import timeit
from random import randint

from pygame.locals import *

from car import Auto
from mouse import Mouse
from primitives import *
from track import RaceTrack
from viewer import TargetViewer

pygame.init()

reset_timer = 30000  # reset the track after n ms
num_car = 200  # how many car to spawn


class AutoSimulation:
    def __init__(self):
        self.viewer = TargetViewer(self)
        self.mouse = Mouse(self)
        self.text = Text("Generation : 0")
        self.autos = []
        self.top_auto = None
        self.track = RaceTrack(self)
        self.learning_rate = .1
        self.generation = 0
        self.events = []
        self.dt = 0
        self.start_time = pygame.time.get_ticks()
        self.addNewAutos(num_car)

    def generateTrack(self, difficulty):
        size = randint(int(800 - (400 * difficulty)), 800)
        width = size / 4
        self.track.generateTrack(Rect((size, size), (-size / 2.0, -size / 2.0)), width)

    def addNewAutos(self, n):
        for i in range(n):
            self.autos.append(Auto(self))

    def getFittestAuto(self):
        max_fit = -1
        top_autos = []
        for auto in self.autos:
            auto.top = 0
            if auto.fitness > max_fit:
                max_fit = auto.fitness
                top_autos = [auto]
            if not auto.stop:
                pass
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
                    car.stop = 1
        self.getFittestAuto()
        self.mouse.update()

        if not self.top_auto.stop:
            self.viewer.updatePos(self.top_auto.body.pos)
        if pygame.time.get_ticks() - self.start_time > reset_timer:
            self.reset()

    def cont(self):
        stopped = 0
        for car in self.autos:
            stopped += car.stop
        all_stopped = stopped == (len(self.autos) - 1)  # KERU
        return 1 - all_stopped

    def end(self):
        new_batch = []
        # count car that are still running and send them for the next run as-is
        running_car_count = 0
        for running_auto in self.autos:
            if running_auto.stop == 0:
                running_car_count = running_car_count + 1
                # new_batch.append(running_auto.makeChild())    # you can send send mutated child instead too
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
            if not car.stop:
                car.draw()  # only draw car that are still running

        self.top_auto.draw()
        self.top_auto.brain.draw(Rect((200, 150), (self.viewer.pos.x + 50, self.viewer.pos.y + 50)))
        self.track.draw()
        self.mouse.draw()
        self.text.draw(self.viewer.display)
        self.top_auto.brain.draw(Rect((200, 150), (self.viewer.pos.x + 50, self.viewer.pos.y + 50)))
        #self.viewer.draw(self.top_auto.inputs)     #enable to see sensor
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
        self.world.dt = 0.01  # make the simulation independent of framerate
        # self.world.dt = (timeit.default_timer() - self.t)  # make simulation dependent of framerate
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
