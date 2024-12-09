# -*- coding: utf-8 -*-
"""
@author: cub525
"""

from manim import *
import numpy as np
import cmath
# from fractal import Fractal, LevyC, Terdragon, HeighwayDragon
import fractal
import matplotlib.colors as cl


colors = [i for i in cl.TABLEAU_COLORS.values()]


def lines_from_complex(
        s: np.array, ax, stroke_width=8, color=("#0A68EF", "#4AF1F2", "#0A68EF")):
    line_group = VGroup(*[
        Line(ax.c2p(np.real(s[i]), np.imag(s[i])),
             ax.c2p(np.real(s[i + 1]), np.imag(s[i + 1])))
        for i in range(0, len(s) - 1, 2)])
    KC = (
        VMobject(stroke_width=stroke_width)
        .set_points(line_group.get_all_points())
        .set_color(color)
    )
    return KC


class AnimFractal(Scene):
    divide_first = False

    def construct(self):

        level = Variable(0, Tex("level"), var_type=Integer) \
            .set_color("#4AF1F2")
        txt = (
            VGroup(Tex(self.name, font_size=60), level)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UL)
        )
        xlim, ylim = self.fractal.xlim, self.fractal.ylim
        ratio = ( xlim[1] - xlim[0] )/( ylim[1] - ylim[0] )
        ax = Axes(
            x_range=xlim, y_range=ylim, x_length=6*ratio, y_length=6
            )
        manim_objs = [lines_from_complex(
            points, ax, stroke_width=2,
            color=colors[i])
            for i, points in enumerate(self.fractal._plot_list)
            ]
        if self.generate_image:
            if self.divide:
                self.fractal.divided_iterate(self.iterations)
            else:
                self.fractal.iterate(self.iterations)
                if self.tile:
                    self.fractal.tile()
            manim_objs = [lines_from_complex(
                self.fractal._plot_list[i], ax, stroke_width=1, color=colors[i])
                for i in range(len(self.fractal._plot_list))]
            level.traker.set_value(self.iterations)
            self.add(txt, *manim_objs)
            self.wait()
        elif self.tile and not self.divide:
            self.add(txt, *manim_objs)
            for i in range(self.iterations):
                self.fractal.iterate()
                self.fractal.tile()
                # self.add(txt, *manim_objs)
                self.play(
                    level.tracker.animate.set_value(i))
                if len(manim_objs) != len(self.fractal._plot_list):
                    manim_objs.extend([manim_objs[0].copy()
                                       for i in range(len(self.fractal._plot_list) - len(manim_objs))])
                for j, points in enumerate(self.fractal._plot_list):
                    self.play(manim_objs[j].animate.become(lines_from_complex(
                        points, ax, stroke_width=2,
                        color=colors[j])), run_time = (1/(1 + i)))
                # self.fractal.iterate()
                self.wait(1/(1 + i))
            self.wait()
        else:
            self.add(txt, *manim_objs)
            for i in range(self.iterations):
                if i == self.iterations - 1 and not self.divide_first:
                    self.fractal.divided_iterate()
                elif self.divide_first:
                    self.fractal.divided_iterate()
                else:
                    self.fractal.iterate()
                self.add(manim_objs[-1])
                self.play(
                    level.tracker.animate.set_value(i))
                if len(manim_objs) != len(self.fractal._plot_list):
                    manim_objs.extend([manim_objs[0].copy()
                                       for i in range(len(self.fractal._plot_list) - len(manim_objs))])
                for j, points in enumerate(self.fractal._plot_list):
                    self.play(manim_objs[j].animate.become(lines_from_complex(
                        points, ax, stroke_width=2,
                        color=colors[j])), run_time = (1/(1 + i))),
                self.wait(1/(1 + i))
            self.wait(1)


class HeighwayDragon(AnimFractal):
    def setup(self):
        self.fractal = fractal.HeighwayDragon()
        self.iterations = 15
        self.name = "Heighway Dragon"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = True
        self.divide_first = True


class TwinDragon(AnimFractal):
    def setup(self):
        self.fractal = fractal.TwinDragon()
        self.iterations = 12
        self.name = "Twin Dragon"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = True
        self.divide_first = True


class GoldenDragon(AnimFractal):
    def setup(self):
        self.fractal = fractal.GoldenDragon()
        self.iterations = 15
        self.name = "Golden Dragon"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = True
        self.divide_first = True


class Terdragon(AnimFractal):
    def setup(self):
        self.fractal = fractal.Terdragon()
        self.iterations = 9
        self.name = "Terdragon"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = True
        self.divide_first = False


class Fudgeflake(AnimFractal):
    def setup(self):
        self.fractal = fractal.FudgeFlake()
        self.iterations = 8
        self.name = "Fudgeflake"
        self.generate_image: bool = False
        self.tile: bool = True
        self.divide: bool = False


class LevyC(AnimFractal):
    def setup(self):
        self.fractal = fractal.LevyC()
        self.iterations = 15
        self.name = "Levy C"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = True
        self.divide_first = False

class KochFlake(AnimFractal):
    def setup(self):
        self.fractal = fractal.KochFlake()
        self.iterations = 5
        self.name = "Koch Flake"
        self.generate_image: bool = False
        self.tile: bool = True
        self.divide: bool = False

class Kochawave(AnimFractal):
    def setup(self):
        self.fractal = fractal.Kochawave()
        self.iterations = 8
        self.name = "Kochawave"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = False
        self.divide_first = False


class Pentadendrite(AnimFractal):
    def setup(self):
        self.fractal = fractal.Pentadendrite()
        self.iterations = 5
        self.name = "Pentadendrite"
        self.generate_image: bool = False
        self.tile: bool = True
        self.divide: bool = False



class Pentigree(AnimFractal):
    def setup(self):
        self.fractal = fractal.Pentigree()
        self.iterations = 5
        self.name = "Pentigree"
        self.generate_image: bool = False
        self.tile: bool = True
        self.divide: bool = False

class DurerPentagon(AnimFractal):
    def setup(self):
        self.fractal = fractal.DurerPentagon()
        self.iterations = 6
        self.name = "Durer Pentagon"
        self.generate_image: bool = False
        self.tile: bool = True
        self.divide: bool = False


class Flowsnake(AnimFractal):
    def setup(self):
        self.fractal = fractal.Flowsnake()
        self.iterations = 5
        self.name = "Flowsnake"
        self.generate_image: bool = False
        self.tile: bool = False
        self.divide: bool = True
        self.divide_first = True


class UnivVideo(Scene):
    def construct(self):
        # Display a centered title Fractals with a subtitle Univ 101 Final Project in small font
        title = Text("Fractals", font_size=60).to_edge(UP, buff=1.0)
        subtitle = Text("Univ 101 Final Project", font_size=20).next_to(title, DOWN)
        self.play(
            Write(title),
            Write(subtitle)
        )
        self.wait()
        by = Text("by", font_size=20).to_edge(LEFT, buff=.5)
        # Add the emmett.png image to the scene
        name = ImageMobject("assets/images/emmett.png").scale(2.0).next_to(by, RIGHT)
        self.play(
            Write(by),
            FadeIn(name, scale=.2)
        )
        self.wait(2)
# %manim -p -ql -v Warning UnivVideo

class Outro(Scene):
    # Simple scene that says thank you for watching, hope you enjoyed the
    # fractal in blue text that lasts for 8 seconds
    def construct(self):
        text = Text("Thank you for watching!\n", font_size=60, color=BLUE).to_edge(UP)
        text2 = Text("Hope you enjoyed the fractals", font_size=60, color=BLUE).next_to(text, DOWN)
        self.play(Write(text))
        self.wait(2)
        self.play(Write(text2))
        self.wait(2)
        # Add have a great next semester in smaller green font below
        text2 = Text("Have a great next semester", font_size=40, color=GREEN).next_to(text2, DOWN)
        self.play(Write(text2))
        self.wait(3)
