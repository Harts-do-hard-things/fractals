# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:37:09 2021

@author: cub525
"""

from ply import lex, yacc
import sys
import numpy as np

try:
    from matrixfractal import IFSystem
except ImportError:
    IFSystem = object

class obj_container(object):
    def __init__(self, object_dict):
        self.__dict__.update(object_dict)


states = (("inside", "exclusive"),)

tokens = [
    "ID",
    "ARRAY"
]

t_ANY_ignore = " \t"


def t_3D(t):
    r"\(3D\)"
    pass

def t_ANY_newline(t):
    r"\n"
    t.lexer.lineno += 1
    # return t


def t_ANY_COMMENT(t):
    r";.*"
    pass


def t_ANY_error(t):
    # raise NameError(f"name '{t.value}' is not defined")
    # print(t)
    t.lexer.skip(1)


t_ID = "\w+"


def t_LBRACKET(t):
    "{"
    t.lexer.begin("inside")

def t_inside_LBRACKET(t):
    "{"
    print("missing }")
    t_ANY_error(t)

def t_inside_RBRACKET(t):
    "}"
    t.lexer.begin("INITIAL")


def t_inside_eof(t):
    raise EOFError("EOF while scanning 2D Array: missing '}'")


def t_inside_ARRAY(t):
    r"(?:-?(?:\d+|\d*\.\d*)[^\n;}]*)"
    t.value = np.fromstring(t.value, sep=' ')
    return t


# Begin Parsing Rules
def p_calc(p):
    """
    calc : obj_list
         | expression
         | none
    """
    p[0] = p[1]
    if p[1] in ("exit", "quit"):
        sys.exit()
    elif type(p[1]) == list:
        global objs
        objs = obj_container({"IFS_" + o.__name__: o for o in p[0]})


def p_obj_list_1(p):
    """ obj_list : object"""
    p[0] = [p[1]]


def p_obj_list_2(p):
    """ obj_list : obj_list object"""
    p[0] = p[1] + [p[2]]


def p_object(p):
    """ object : Name 2D_ARRAY """
    p[0] = type(p[1], (IFSystem,), {"eq": np.array(p[2])})

def p_name(p):
    """ Name : ID"""
    p[0] = p[1]

def p_name_1(p):
    """ Name : Name ID"""
    p[0] = p[1] + p[2]

def p_2D_ARRAY_1(p):
    """2D_ARRAY : ARRAY"""
    p[0] = [p[1]]


def p_2D_ARRAY_2(p):
    """2D_ARRAY : 2D_ARRAY ARRAY"""
    p[0] = p[1] + [p[2]]


def p_expression(p):
    """
    expression : ID
               | ARRAY
    """
    p[0] = p[1]


def p_none(p):
    """
    none :
    """
    pass


def p_error(p):
    print("ERROR")
    print(p)
    # sys.exit()


def interpret_file(path):
    with open(path, 'r') as f:
        string = f.read()
    parser.parse(string)
    return objs

lexer = lex.lex()
parser = yacc.yacc(debug=True)    

# if __name__ == "ifslex":

if __name__ == "__main__":
    test_string = """my binary { ; comment allowed here
  ; and here
  .5  .0 .0 .5 -2.563477 -0.000003 .333333   ; also comment allowed here
  .5  .0 .0 .5  2.436544 -0.000003 .333333
  .0 -.5 .5 .0  4.873085  7.563492 .333333
  }
"""
    # lexer.input(test_string)
    # while True:
    #     tok = lexer.token()
    #     if not tok:
    #         break
    #     print(tok)
    # while True:
    #     try:
    #         s = input(">> ")
    #     except (EOFError, KeyboardInterrupt):
    #         break
    #     parser.parse(s)
    parser.parse(test_string)
    # interpret_file("fractint.ifs")
