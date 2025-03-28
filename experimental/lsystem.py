# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:13:44 2021

@author: cub525
"""

import dataclasses as dtcls
import turtle



@dtcls.dataclass()
class L_System:
    
    axiom: str
    angle: float
    rules: dict[str, str]
    def __post_init__(self):
        self.commands = {
        'F': lambda t:t.forward(20),
        'G': lambda t:t.forward(20),
        'Z': lambda _: None,
        '+': lambda t:t.right(self.angle),
        '-': lambda t:t.left(self.angle)
        # '−': lambda t:t.left(self.angle)
        }
        self.rules = str.maketrans(self.rules)
        self.state = self.axiom

    def iterate(self, i: int) -> None:
        for _ in range(i):
            self.state = self.state.translate(self.rules)
    
    def plot(self):
        t = turtle.Turtle()
        # turtle.tracer(0, 0)
        for c in self.state:
            if c in self.commands:
                self.commands[c](t)
        # turtle.update()
        turtle.mainloop()
        turtle.bye()

    def __str__(self):
        n1='\n'
        return f"""Angle {self.angle}
Axiom {self.axiom}
{''.join([f'{chr(key)} --> {self.rules[key]}' + n1 for key in self.rules])}"""

    @classmethod
    def from_input(cls, string = None):
        if not string:
            string = input("Paste the fractal information:")
        # string.translate(str.maketrans({'−':'-'}))
        info_list = string.splitlines()
        angle = int(info_list[0][5:])
        axiom = info_list[1][6:]
        r = {i[0]:i[3:] for i in info_list[2:]}
        f = cls(axiom,angle,r)
        return f


if __name__ == '__main__':
    flowsnake = L_System('F',70,{'F':"+F--F+"}) #L_System.from_input("""Angle 60
    # Axiom F
    # F->F-G--G+F++FF+G-
    # G->+F-GG--G-F++F+G""")
    flowsnake.iterate(7)
    flowsnake.plot()
