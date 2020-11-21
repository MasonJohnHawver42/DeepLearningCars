from vector import Vector


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
