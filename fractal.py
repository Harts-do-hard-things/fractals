# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:14:22 2016

@author: Harts
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import cmath


class Fractal:
    def __init__(self,S0,func_list):
        self.S0 = S0
        self.S  = S0
        self.func_list = func_list
        self.i = 0
    
    def iterate(self, i):
        for j in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func,self.S)))
                S.append(float('nan'))
            self.S = S

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot = ax.plot(np.real(self.S),np.imag(self.S))            
        plt.axis('equal')

SF =[]
S0 = [0,1]

#Vars Used in terdragon    
lamda = .5 - 1j/2/math.sqrt(3)
lamdaconj = .5 + 1j/2/math.sqrt(3)

#Vars used in twindragon
S0_twin = [0,1,1-1j]

#Vars used in golden dragon

phi = (1 +math.sqrt(5))*.5
r = (1/phi)**(1/phi)
A = math.acos((1+r**2-r**4)/2/r)
B = math.acos((1-r**2+r**4)/2/r**2)

#Vars used in Pentigree
P_r = (3-math.sqrt(5))*.5
P_a1 = (math.cos(.2*math.pi) + 1j*math.sin(.2*math.pi))
P_a2 = (math.cos(.6*math.pi) + 1j*math.sin(.6*math.pi))
P_na1 = (1*math.cos(.2*math.pi) + -1j*math.sin(.2*math.pi))
P_na2 = (1*math.cos(.6*math.pi) + -1j*math.sin(.6*math.pi))

#Vars used in pentadendrite
Dr = math.sqrt((6-math.sqrt(5))/31)
RA = 0.20627323391771757578747269015392
PA = (1*math.cos(RA) + 1j*math.sin(RA))
SA = (math.pi*0.4)
BA = (1*math.cos(RA + SA) + 1j*math.sin(RA + SA))
CA = (1*math.cos(RA - SA) + 1j*math.sin(RA - SA))
DA = (1*math.cos(RA - 2*SA) + 1j*math.sin(RA - 2*SA))

#Vars used in koch snowflake
K0 = (0+1j)
Ka = (.5 + .5*1j*math.sqrt(3))
Kna = (.5 - .5*1j*math.sqrt(3))
kr = 1/3
SK = [-1,0]

# Plain Ole' Dragon Curve
def dragon1(z):
    return 0.5*(1+1j)*z
    
def dragon2(z):
    return 1-0.5*(1-1j)*z

# Levy C
def func1(z):
    return 0.5*(1-1j)*z
    
def func2(z):
    return 1+0.5*(1+1j)*(z-1)

# Twin Dragon
def twin1(z):
    return 0.5*(1+1j)*z
    
def twin2(z):
    return 1-0.5*(1+1j)*z

#Terdragon xtreme
def ter1(z):
    return lamda*z
def ter2(z):
    return 1j/math.sqrt(3)*z+lamda
def ter3(z):
    return lamda*z + lamdaconj

#golden dragon

def gd1(z):
    
    return r*z*cmath.exp(A*1j)
    
def gd2(z):
    return r**2*z*cmath.exp((math.pi-B)*1j)+1

#z2 golden dragon

def z2gd1(z):
    return gd1(z)

def z2gd2(z):
    return r*z*cmath.exp((A-math.pi)*1j)
    
def z2gd3(z):
    return gd2(z)  
    
def z2gd4(z):
    return r**2*z*cmath.exp(-B*1j)-1    

#z2 levy curve

def z2levy1(z):
    return func1(z)

def z2levy2(z):
    return -(1+0.5*(1+1j)*(z-1))
    
def z2levy3(z):
    return func2(z)  
    
def z2levy4(z):
    return -(0.5*(1-1j)*z)

#z2 dragon curve

def z2dragon1(z):
    return dragon1(z)

def z2dragon2(z):
    return -(1-0.5*(1-1j)*z)
    
def z2dragon3(z):
    return dragon2(z)  
    
def z2dragon4(z):
    return -(0.5*(1+1j)*z)

#Pentigree

def pent1(z):
    return P_r*P_a1*z

def pent2(z):
    return P_r*P_a2*z + P_r*(P_a1)

def pent3(z):
    return P_r*P_na1*z + P_r*(P_a1 + P_a2)

def pent4(z):
    return P_r*P_na2*z + P_r*(P_a1 + P_a2 + P_na1)

def pent5(z):
    return P_r*P_na1*z + P_r*(P_a1 + P_a2 + P_na1 + P_na2)

def pent6(z):
    return P_r*P_a1*z + P_r*(P_a1 + P_a2 + P_na1 + P_na2 + P_na1)

#Pentadendrite

def pend1(z):
    return Dr*(PA*z)

def pend2(z):
    return Dr*(BA*z+PA)

def pend3(z):
    return Dr*(PA*z + PA + BA)

def pend4(z):
    return Dr*(DA*z + 2*PA + BA)

def pend5(z):
    return Dr*(CA*z + DA + 2*PA + BA)

def pend6(z):
    return Dr*(PA*z + 2*PA + BA + CA + DA)

#Koch Snowflake

def koch1(z):
    return kr*(z)

def koch2(z):
    return kr*(Ka*z + 1 )

def koch3(z):
    return kr*(Kna*z + 1 + Ka)

def koch4(z):
    return kr*(z + 2)


# Builds a unit-pentagon in complex plane
star = np.exp(1j*np.arange(0,361,72)*math.pi/180)
pentagon = np.cumsum(star)


if __name__ == "__main__":
#    dragon = Fractal(S0,[dragon1,dragon2])
#    dragon.iterate(1)
#    dragon.plot()
#    levy = Fractal(S0,[func1,func2])
#    levy.iterate(17)
#    levy.plot()
#    z2_dragon = Fractal(S0,[z2dragon1,z2dragon2,z2dragon3,z2dragon4])
#    z2_dragon.iterate(10)
#    z2_dragon.plot()
#    z2levy = Fractal([0,1],[func1,z2levy2,func2,z2levy4])
#    z2levy.iterate(10)
#    z2levy.plot()
#    twin_dragon = Fractal(S0_twin,[twin1,twin2])
#    twin_dragon.iterate(17)
#    twin_dragon.plot()
#    terdragon = Fractal(S0,[ter1,ter2,ter3])
#    terdragon.iterate(10)
#    terdragon.plot()
#    golden_dragon = Fractal(S0,[gd1,gd2])
#    golden_dragon.iterate(14)
#    golden_dragon.plot()
#    z2_golden_dragon = Fractal(S0,[z2gd1,z2gd2,z2gd3,z2gd4])
#    z2_golden_dragon.iterate(10)
#    z2_golden_dragon.plot()
#    pentigree = Fractal(S0,[pent1,pent2,pent3,pent4,pent5,pent6])
#    pentigree.iterate(1)
#    pentigree.plot()
    pentadentrite = Fractal([0,1],[pend1,pend2,pend3,pend4,pend5,pend6])
    pentadentrite.iterate(2)
    pentadentrite.plot()
#    Koch_fake = Fractal(SK,[koch1,koch2,koch3,koch4])
#    Koch_fake.iterate(1)
#    Koch_fake.plot()