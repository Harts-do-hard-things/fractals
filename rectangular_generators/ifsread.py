# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:37:09 2021

@author: cub525
"""

from ply import lex, yacc
import sys
import numpy as np
from matrixfractal import IFSystemRand
from dotmap import DotMap


ifs = DotMap()

states = (("inside", "exclusive"),
          ("array", "inclusive"))

tokens = [
    "ID",
    "ARRAY",
    "DOCS"
]

literals = [
    '{', '}']

t_ANY_ignore = " \t"


def t_3D(t):
    r"\(3D\)"
    pass


def t_ANY_newline(t):
    r"\n"
    t.lexer.lineno += 1


def t_inside_DOCS(t):
    r";\s*(.*)"
    t.value = t.value[1:]
    return t


def t_array_COMMENT(t):
    r";.*"
    pass


def t_ANY_error(t):
    raise NameError(f"name '{t.value}' is not defined")


t_ID = r"\w+"


def t_LBRACKET(t):
    "{"
    t.type = '{'
    t.lexer.begin("inside")


def t_inside_LBRACKET(t):
    "{"
    t.type = '{'
    raise SyntaxError("Nested 2D Array: missing '}'")


def t_inside_array_RBRACKET(t):
    "}"
    t.type = '}'
    t.lexer.begin("INITIAL")


def t_inside_array_eof(t):
    raise EOFError("EOF while scanning 2D Array: missing '}'")


def t_inside_array_ARRAY(t):
    r"(?:-?(?:\d+|\d*\.\d*)[^\n;}]*)"
    t.value = np.fromstring(t.value, sep=' ')
    t.lexer.begin("array")
    return t


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
        ifs.update({"IFS_" + o.__name__: o for o in p[0]})


def p_obj_list_1(p):
    """ obj_list : object"""
    p[0] = [p[1]]


def p_obj_list_2(p):
    """ obj_list : obj_list object"""
    p[0] = p[1] + [p[2]]


def p_object(p):
    """ object : Name 2D_ARRAY"""
    p[0] = type(p[1], (IFSystemRand,), {"eq": np.array(p[2])})


def p_object_1(p):
    """ object : Name docs 2D_ARRAY"""
    p[0] = type(p[1], (IFSystemRand,), {"eq": np.array(p[3]), "__doc__": p[2]})


def p_name(p):
    """ Name : ID"""
    p[0] = p[1]


def p_name_1(p):
    """ Name : Name ID"""
    p[0] = p[1] + p[2]


# grab docstrings, two rules and combine them with newlines
def p_docs(p):
    """ docs : DOCS"""
    p[0] = p[1]


def p_docs_1(p):
    """ docs : docs DOCS"""
    p[0] = p[1] + '\n' + p[2]


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
               | DOCS
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


def interpret_file(path):
    with open(path, 'r') as f:
        string = f.read()
    parser.parse(string)


lexer = lex.lex()
parser = yacc.yacc()
interpret_file("ifs_data\\fractint.ifs")
