# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:02:10 2021

@author: janih
"""

from fractal import *
Fractal.translate.__doc__= """
Translates the fractal by offset, then rotates it by angle (rads)
In that Order. Is used in Fudgeflake to tile the terdragon around an
equilateral triangle. Appends the translated S onto plot_list

.. code-block:: python
   # Fudgeflake tile method
   def tile(self):
       self.translate(0, math.pi/3)
       self.translate(1, 2*math.pi/3)


Parameters
----------
offset : complex
    The complex to offset the fractal.
angle : float
    The angle, in radians, that rotates the fractal

Returns
-------
None

"""

Fractal.tile.__doc__ = """
Empty method called by plot; intended to be filled with calls of the
translate method in child classes

Returns
-------
None.

"""

Fractal.iterate.__doc__ = """
clears the plot list (ensures proper gif plotting)
maps the functions to S, reassigning S
appends S back to plot list

Parameters
----------
i : int
    The number of iterations to advance from the current state

Returns
-------
None

"""

Fractal.__doc__ = """
A class used to draw and calculate Fractals.

Based on research by
`Larry Riddle <https://larryriddle.agnesscott.org/ifs/ifs.htm>`_

Attributes
----------
S0 : List[complex]
    the initial starting points, on the complex plane

S : List[complex] = S0
    the current points of the fractal

func_list : List[functions]
    the functions to be applied every iteration

plot_list List[S]:
    the list to be plotted (note: most of the list is generated by the plot method)

plot_handle : List[plot objects]
    the matplotlib plot objects generated after plotting

Methods
-------
iterate(i: int)
    maps the list of functions to S, overwriting S

plot()
    calls tile() before plotting all S in plot_list.

tile()
    empty method designed to be filled with calls of translate

translate(offset: complex, angle: float)
    transforms the fractal by the offset, then rotates the fractal by angle

save_gif(iterations: int, duration: int = 1000)
saves a gif at '__name__ _iterations.gif' with a frame duration of duration milliseconds

"""


DragonFractal.__doc__ = """
.. _dragon-fractal:
A class used to draw and calculate Fractals that require nans inbetween segments

Based loosely on research by
`Larry Riddle <https://larryriddle.agnesscott.org/ifs/ifs.htm>`_

Attributes
----------
S0 : List[complex]
    the initial starting points, on the complex plane

S : List[complex] = S0
    the current points of the fractal

func_list : List[functions]
    the functions to be applied every iteration

plot_list List[S]:
    the list to be plotted (note: most of the list is generated by the plot method)

plot_handle : List[plot objects]
    the matplotlib plot objects generated after plotting

Methods
-------
**segment()** 
    Modifies S so that it is 'segmented' e.g. for every line segment there is a nan
    Allows for plotting Different Fractals that don't have a specific set of functions

iterate(i: int)
    maps the list of functions to S, overwriting S

plot()
    calls tile() before plotting all S in plot_list.

tile()
    empty method designed to be filled with calls of translate

translate(offset: complex, angle: float)
    transforms the fractal by the offset, then rotates the fractal by angle

save_gif(iterations: int, duration: int = 1000)
saves a gif at '__name__ _iterations.gif' with a frame duration of duration milliseconds

"""

BinaryTree.__doc__ = """
Creates Binary Trees Based off of
`Larry Riddle's Webpage <https://larryriddle.agnesscott.org/ifs/pythagorean/symbinarytree.htm>`_

See Documentation for :class:`~fractal.DragonFractal` for implementation details
and Attached Resource for Creation Details
"""

HeighwayDragon.__doc__ = """`Heighway Dragon <https://larryriddle.agnesscott.org/ifs/heighway/heighway.htm>`_ Fractal inheriting from :class:`~fractal.DragonFractal`"""

TwinDragon.__doc__ = """`Twin Dragon <https://larryriddle.agnesscott.org/ifs/heighway/twindragon.htm>`_ Fractal inheriting from :class:`~fractal.DragonFractal`"""

GoldenDragon.__doc__ = """`Golden Dragon <https://larryriddle.agnesscott.org/ifs/heighway/goldenDragon.htm>`_ Fractal inheriting from :class:`~fractal.DragonFractal`"""

Terdragon.__doc__ = """`Terdragon <https://larryriddle.agnesscott.org/ifs/heighway/terdragon.htm>`_ Fractal inheriting from :class:`~fractal.Fractal`"""

FudgeFlake.__doc__ = """`Fudgeflake <https://larryriddle.agnesscott.org/ifs/heighway/fudgeflake.htm>`_ Fractal inheriting from :class:`fractal.Terdragon`"""

LevyC.__doc__ = """`Levy C Curve <https://larryriddle.agnesscott.org/ifs/levy/levy.htm>`_ inheriting from :class:`~fractal.Fractal`"""

LevyTapestryOutside.__doc__ = """`Levy Tapestry <https://larryriddle.agnesscott.org/ifs/levy/tapestryOutside.htm>`_ inheriting from :class:`~fractal.LevyC`"""

LevyTapestryInside.__doc__ = """`Levy Tapestry <https://larryriddle.agnesscott.org/ifs/levy/tapestryInside.htm>`_ inheriting from :class:`~fractal.LevyC`"""

KochFlake.__doc__ = """
`Koch Flake <https://larryriddle.agnesscott.org/ifs/kcurve/kcurve.htm>`_ inheriting from :class:`~fractal.Fractal`

Note: this is constructed as a koch curve, then tiled.
"""

Pentadendrite.__doc__ = """`Pentadendrite <https://larryriddle.agnesscott.org/ifs/pentaden/penta.htm>`_ inheriting from :class:`~fractal.Fractal` """






