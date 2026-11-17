"""Palette-defined vocabulary: no test-dependent class discovery."""
from itertools import combinations

COLORS=['하양','노랑','주황','분홍','빨강','갈색','연두','초록','청록','파랑','남색','자주','보라','회색','검정']
SHAPES=['원형','타원형','장방형','반원형','삼각형','사각형','마름모형','오각형','육각형','팔각형','기타']
COLOR_SETS=[(i,) for i in range(len(COLORS))]+list(combinations(range(len(COLORS)),2))
COLOR_NAMES=['+'.join(COLORS[i] for i in s) for s in COLOR_SETS]
COLOR_INDEX={s:i for i,s in enumerate(COLOR_SETS)}


def color_target(bits):
    if len(bits)!=len(COLORS) or any(v not in (0,1,False,True) for v in bits):
        raise ValueError('Expected binary base-color labels')
    key=tuple(i for i,v in enumerate(bits) if v)
    if key not in COLOR_INDEX:
        raise ValueError('Known color must contain exactly one or two colors')
    return COLOR_INDEX[key]


def color_bits(index):
    active=set(COLOR_SETS[index])
    return [int(i in active) for i in range(len(COLORS))]
