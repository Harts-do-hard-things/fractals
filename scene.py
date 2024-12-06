# -*- coding: utf-8 -*-
"""
@author: cub525
"""

from manim import *
import numpy as np
import cmath
from fractal import Fractal, LevyC, Terdragon, HeighwayDragon
import fractal



def lines_from_complex(
        s: np.array, stroke_width=8, color=("#0A68EF", "#4AF1F2", "#0A68EF")):
    line_group = VGroup(*[
        Line((np.real(s[i]), np.imag(s[i]), 0),
             (np.real(s[i + 1]), np.imag(s[i + 1]),
              0)) for i in range(0, len(s) - 1, 2)])
    KC = (
        VMobject(stroke_width=stroke_width)
        .set_points(line_group.get_all_points())
        .set_color(color)
    )
    return KC


class CW_Levy_C(Scene):
    def construct(self):

        def lines_from_complex(
                s: np.array, stroke_width=8, color=("#0A68EF", "#4AF1F2", "#0A68EF")):
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

        level = Variable(0, Tex("level"), var_type=Integer) \
            .set_color("#4AF1F2")
        txt = (
            VGroup(Tex("GoldenDragon", font_size=60), level)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UL)
        )
        anim_fractal = fractal.GoldenDragon()
        ax = Axes(x_range=anim_fractal.limits[:2], y_range=anim_fractal.limits[2:], y_length=6).to_edge(DOWN)
        iterations = 15
        # anim_fractal.iterate(iterations)
        # anim_fractal.translate_in_place(-.5, 0)
        # anim_fractal.scale(5)
        manim_obj = lines_from_complex(
            anim_fractal._plot_list[0], stroke_width=2,
            color=(RED_B, RED_C, RED_B)).to_edge(DOWN, buff=2.0)

        self.add(txt, manim_obj)
        self.wait()


        for i in range(iterations):
            anim_fractal.iterate()
            # anim_fractal.translate_in_place(-.5, 0)
            # anim_fractal.scale(5)
            self.play(
                level.tracker.animate.set_value(i),
                manim_obj.animate.become(lines_from_complex(
                    anim_fractal._plot_list[0], stroke_width=2,
                    color=(RED_B, RED_C, RED_B))
                    .to_edge(DOWN, buff=2.0)
                ),
            )
            self.wait()



        #Make a new object and display rotating it to make a levy c tapestry
        # self.play(
        #     manim_obj.animate.scale(.5).to_edge(DOWN, buff=2.0)
        # )
        # manim_obj2 = manim_obj.copy().set_color(RED)
        # center_of_s0 = (manim_obj.get_anchors()[0] + manim_obj.get_anchors()[-1]) / 2
        # self.play(
        #     Rotate(manim_obj2, angle=PI, about_point=center_of_s0)
        # )
        self.wait()

class Levy_C_Image(Scene):
    def construct(self):
        anim_fractal = LevyC()
        # set bounds
        ax = Axes(x_range=anim_fractal.limits[:2], y_range=anim_fractal.limits[2:], x_length=8).to_edge(DOWN)
        def lines_from_complex(
                s: np.array, stroke_width=8, color=("#0A68EF", "#4AF1F2", "#0A68EF")):
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
        iterations = 18
        anim_fractal.iterate(iterations)
        manim_obj = lines_from_complex(
            anim_fractal._plot_list[-1], stroke_width=1, color=WHITE)
        self.add(manim_obj)
        self.wait()


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
        name = ImageMobject("assets/images/emmett1.png").scale(2.0).next_to(by, RIGHT)
        self.play(
            Write(by),
            FadeIn(name, scale=.2)
        )
        self.wait(2)
        # self.se

# %manim -p -ql -v Warning UnivVideo