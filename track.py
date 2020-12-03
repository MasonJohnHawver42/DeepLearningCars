from random import randint, choice

import numpy as np
from scipy.spatial import ConvexHull

from primitives import *


class Track:
    def __init__(self, start_outer=(0, 0), start_inner=(0, 0),
                 end_outer=(0, 0), end_inner=(0, 0), next_track=None) -> None:
        self.track_outer = Line(start_outer, end_outer)
        self.track_inner = Line(start_inner, end_inner)
        self.gate = Line(start_outer, start_inner)
        self.next_track = next_track
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
        self.dir = choice([-1,1])    # (randint(0, 1) * 2) - 1

    def generateTrack(self, bounding_rect, width) -> None:
        '''width is the desired width of the track'''
        self.tracks = []

        track_outer = Poly()
        track_inner = Poly()

        track_outer.points = []
        track_inner.points = []

        points = np.random.rand(randint(20, 30), 2)
        hull = ConvexHull(points)
        self.dir = choice([-1,1])   # (randint(0, 1) * 2) - 1
        verts = hull.vertices  # vertices is the list of the index of the point making the convex hull

        for vert in verts:
            distance = 100     # distance ?
            point = points[vert]    # the (x,y) coordinate of the convex hull point
            point -= [.5, .5]       # the hull is in [0, 1], make it [-0.5,0.5] so [0,0] is ~ the center

            point = (point * [bounding_rect.size.x, bounding_rect.size.y])  # multiply into pixel coordinate
            point = Vector(point[0], point[1])  #make a vector from it

            for p in track_outer.points:
                distance = Vector()
                distance.set(point)
                distance.sub(p)
                distance = distance.getMag()
                if distance < 80:
                    break

            if distance > 80:
                track_outer.points.append(point)
                track_inner.points.append(Vector(point[0], point[1]))

        track_outer.pos = bounding_rect.getCenter()
        track_inner.pos = bounding_rect.getCenter()

        scale = -1 * ((width / min(bounding_rect.size.x, bounding_rect.size.y)) - 1)
        # scale is the mmultiplier to make the innertrack
        # TODO : it would be nice to have a minimum track width defined somewhere

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
        """draw the track : outer + inner + circles"""
        circles = []
        lines = []

        for track in self.tracks:
            lines.append(track.track_outer)
            lines.append(track.track_inner)
            lines.append(track.gate)

            circles.append(Circle(5, track.track_inner.start, track.color))
            circles.append(Circle(5, track.track_outer.start, track.color))

        self.world.viewer.draw(circles + lines)
