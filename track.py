from random import randint

import numpy as np
from scipy.spatial import ConvexHull

from primitives import *


class Track:
    def __init__(self, so=(0, 0), si=(0, 0), eo=(0, 0), ei=(0, 0), nt=None) -> None:
        self.track_outer = Line(so, eo)
        self.track_inner = Line(si, ei)
        self.gate = Line(so, si)
        self.next_track = nt
        self.color = (0, 0, 0)

    def activate(self) -> None:
        self.color = (0, 0, 255)
        self.next_track.color = (0, 255, 0)

    def deactivate(self) -> None:
        self.color = (0, 0, 0)
        self.next_track.color = (0, 0, 0)

    def getCenter(self) -> Vector:
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

    def generateTrack(self, bounding_rect, width) -> None:
        self.tracks = []

        track_outer = Poly()
        track_inner = Poly()

        track_outer.points = []
        track_inner.points = []

        points = np.random.rand(randint(20, 30), 2)
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

    def draw(self) -> None:
        circles = []
        lines = []  # [Line(self.points[0], self.points[-1])]

        for track in self.tracks:
            lines.append(track.track_outer)
            lines.append(track.track_inner)
            lines.append(track.gate)

            circles.append(Circle(5, track.track_inner.start, track.color))
            circles.append(Circle(5, track.track_outer.start, track.color))

        self.world.viewer.draw(circles + lines)
