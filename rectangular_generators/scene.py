from manim import *
from ifsread import ifs

import math
import matplotlib.colors as cl
# config.frame_height = 16
# config.frame_width = 9
# config.pixel_width = 1080
# config.pixel_height = 1920
config.frame_rate = 60
# config.renderer = "opengl"
# config.background_color = '#455D3E'

colors = [i for i in cl.TABLEAU_COLORS.values()]

def divide_dots(dots, dots_before):
    start = math.log(len(dots), dots_before)
    dots_groups = [dots[i] for i in range(dots_before)]
    for i in range(1, int(start)):
        for j in range(1, dots_before):
            dots_groups.append(VGroup(*dots[dots_before ** i * j:dots_before ** (i) * (j + 1)]))
    dots_groups.append(VGroup(*dots[dots_before ** (int(start)):]))
    return dots_groups


class RandomlyDeterminedIFS(Scene):
    def construct(self):
        txt = (
            VGroup(Tex(self.name, font_size=60))
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UL)
        )
        Dot.set_default(radius=0.03, color=WHITE)
        self.fractal.iterate(self.pre_iterations)
        ax = Axes(x_range=self.fractal.xlim, y_range=self.fractal.ylim, x_length=6, y_length=6).to_edge(DOWN)
        def ifs_to_dots(ifs, ax) -> list[Dot]:
            return [Dot(ax.c2p(*i), color=colors[j]) for i,j in zip(ifs.S, ifs.trans_used)]

        dots = ifs_to_dots(self.fractal, ax)
        groups = divide_dots(dots, self.pre_iterations)
        self.add(txt)

        self.wait(0.5)
        self.play(
            AnimationGroup(
                *[FadeIn(d) for d in groups],
                lag_ratio=1,
                run_time=5
            ),
        )

        self.fractal.iterate(self.iterations - self.pre_iterations)
        self.fractal.make_image().save("test.png")
        image = ImageMobject("test.png").move_to(ax.get_center()).scale(6/8)
        self.play(
            FadeOut(*self.mobjects),
            FadeIn(image), lag_ratio=0
            )
        self.add(txt)
        self.wait(1)


class MapleLeaf(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Maple Leaf"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_maple_leaf()


class Fern(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Barnsley's fern"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_fern()


class Coral(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Coral"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_coral()


class Crystal(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Crystal"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_crystal()
        

class Dragon(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Dragon"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_dragon()

class Floor(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Floor"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_floor()

class Spiral(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Spiral"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 3
        self.fractal = ifs.IFS_spiral()

class Triangle(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Sierpinski Triangle"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_triangle()

class Diamond(RandomlyDeterminedIFS):
    def setup(self):
        self.name = "Diamond"
        self.pre_iterations = 2**10
        self.iterations = 1_000_000
        self.divisions = 2
        self.fractal = ifs.IFS_diamond()


# %manim -p -ql --disable_caching -v Warning Fern
