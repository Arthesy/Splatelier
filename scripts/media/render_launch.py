#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Splatelier launch animation renderer.

Renders the Kikyo-palette icon as real anisotropic 2D Gaussians, with each
splat converging from a random initial state (like 3DGS optimisation) and
painting itself onto the palette in order. Outputs PNG frames.

usage: python3 render_launch.py OUT_DIR [--w 1080 --h 1920 --fps 60 --slow 1.4]
"""
import argparse, math, os, random
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--w", type=int, default=1080)
ap.add_argument("--h", type=int, default=1920)
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--dur", type=float, default=2.95)
ap.add_argument("--slow", type=float, default=1.4, help="time stretch (1.0 = original speed); --dur is the length at 1.0")
ap.add_argument("--font", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "InstrumentSerif-Regular.ttf"),
                help="font of the wordmark (Instrument Serif, SIL OFL 1.1; same as the logo)")
ap.add_argument("--no-title", action="store_true", help="leave the wordmark out")
ap.add_argument("--art", type=float, default=0.62, help="palette width / frame width")
ap.add_argument("--only", type=int, default=-1, help="render a single frame index")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

W, H = args.w, args.h
BG = np.array([0x16, 0x1B, 0x46], np.float32) / 255  # launch-screen colour #161B46
IVORY = np.array([0xF5, 0xEC, 0xDD], np.float32) / 255

# icon space (1024) -> frame
K = args.art * W / 690.0           # palette spans x 170..860 in icon space
CX, CY = 515.0, 530.0              # palette centre in icon space
OX, OY = W / 2, H / 2 - 0.02 * H   # optical centre a touch above middle


def to_px(x, y):
    return OX + (x - CX) * K, OY + (y - CY) * K


def hexc(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


# gradient stop profiles (offset, opacity) copied from the SVG icon
P_PETAL = [(0, 1), (.45, .8), (.8, .22), (1, 0)]
P_TIP = [(0, 1), (.45, .75), (.8, .2), (1, 0)]
P_VEIN = [(0, .85), (.6, .35), (1, 0)]
P_WHITE = [(0, .95), (.5, .55), (1, 0)]
P_ACC = [(0, 1), (.45, .7), (.8, .18), (1, 0)]

PURPLE, DEEP, VEIN = hexc("#7C5CE6"), hexc("#5B3FD0"), hexc("#3A2290")
WHITE, STAMEN = hexc("#FFFFFF"), hexc("#F4E7A8")
GREEN, ORANGE, CYAN = hexc("#3E9E68"), hexc("#FFA41F"), hexc("#19B5E8")

# (cx, cy, rx, ry, rot, colour, profile, layer) layer: m=multiply, n=normal
S = []
def add(cx, cy, rx, ry, rot, col, prof, layer, group):
    S.append(dict(cx=cx, cy=cy, rx=rx, ry=ry, rot=rot, col=col, prof=prof, layer=layer, group=group))

add(320, 585, 86, 30, -35, GREEN, P_ACC, "m", 0)
add(365, 640, 62, 22, -10, GREEN, P_ACC, "m", 0)
add(300, 360, 46, 24, -30, CYAN, P_ACC, "m", 0)
add(735, 600, 50, 26, 20, ORANGE, P_ACC, "m", 0)
ANG = [-90, -18, 54, 126, 198]
def ring(r):
    return [(520 + r * math.cos(math.radians(a)), 450 + r * math.sin(math.radians(a)), a) for a in ANG]
for x, y, a in ring(92):  add(x, y, 100, 62, a, PURPLE, P_PETAL, "m", 1)
for x, y, a in ring(158): add(x, y, 48, 22, a, DEEP, P_TIP, "m", 2)
for x, y, a in ring(100): add(x, y, 88, 8, a, VEIN, P_VEIN, "m", 3)
for x, y, a in ring(38):  add(x, y, 36, 18, a, WHITE, P_WHITE, "n", 4)
for x, y, a in ring(14):  add(x, y, 18, 6, a, STAMEN, P_WHITE, "n", 5)

# ---- brush path ---------------------------------------------------------
# keyframes: (time, x, y, painting) in icon space; painting applies to the
# segment that ENDS at the key. Segments use smoothstep easing.
def ringpt(r, a):
    return 520 + r * math.cos(math.radians(a)), 450 + r * math.sin(math.radians(a))
KEYS = [(0.18, 1180, 250, 0), (0.40, 300, 360, 0), (0.47, 288, 352, 1),
        (0.55, 298, 570, 0), (0.65, 340, 600, 1), (0.74, 378, 648, 1)]
t = 0.74
for a in ANG:
    x, y = ringpt(180, a); t += 0.05; KEYS.append((t, x, y, 0))
    x, y = ringpt(34, a);  t += 0.13; KEYS.append((t, x, y, 1))
x, y = ringpt(24, 0); t += 0.05; KEYS.append((t, x, y, 0))
for k in range(1, 9):
    x, y = ringpt(24, k * 45); t += 0.0225; KEYS.append((t, x, y, 1))
t += 0.08; KEYS.append((t, 735, 600, 0))
t += 0.06; KEYS.append((t, 748, 594, 1))
T_EXIT = t + 0.30; KEYS.append((T_EXIT, 1200, 230, 0))


def smooth(x):
    return x * x * (3 - 2 * x)

def brush_at(t):
    if t <= KEYS[0][0]:
        return KEYS[0][1], KEYS[0][2], 0.0
    for (t0, x0, y0, _), (t1, x1, y1, p) in zip(KEYS, KEYS[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            e = u if p else smooth(u)      # strokes are linear, lifts are eased
            return x0 + (x1 - x0) * e, y0 + (y1 - y0) * e, float(p)
    return KEYS[-1][1], KEYS[-1][2], 0.0

# when does the brush tip pass each splat?  -> bloom trigger
TS = np.arange(0, T_EXIT, 1 / 600)
PATH = np.array([brush_at(tt) for tt in TS])
for s in S:
    d = np.hypot(PATH[:, 0] - s["cx"], PATH[:, 1] - s["cy"])
    ok = (PATH[:, 2] > 0) & (d < 32)
    s["t0"] = float(TS[np.argmax(ok)]) if ok.any() else 1.0
    s["dur"] = 0.30

def clamp(x):
    return max(0.0, min(1.0, x))

def ease_out_cubic(x):
    return 1 - (1 - x) ** 3

def ease_out_back(x, s=1.4):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


# ---- palette layers (static, supersampled) -----------------------------
def cubic(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

PAL = [(512, 200), [(720, 200), (860, 330), (860, 500)], [(860, 640), (760, 700), (680, 680)],
       [(610, 662), (590, 720), (620, 780)], [(650, 840), (600, 860), (520, 860)],
       [(320, 860), (170, 720), (170, 530)], [(170, 340), (320, 200), (512, 200)]]

def palette_poly(dy=0.0):
    pts = []
    cur = np.array(PAL[0], float)
    for seg in PAL[1:]:
        c = cubic(cur, np.array(seg[0], float), np.array(seg[1], float), np.array(seg[2], float))
        pts.extend(c.tolist())
        cur = np.array(seg[2], float)
    return [to_px(x, y + dy) for x, y in pts]

SS = 4
def mask_from(draw_fn):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    draw_fn(d)
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255

def ss(poly):
    return [(x * SS, y * SS) for x, y in poly]

shadow_mask = mask_from(lambda d: d.polygon(ss(palette_poly(18)), fill=255))
hx, hy = to_px(470, 690); hr = 52 * K
def pal_draw(d):
    d.polygon(ss(palette_poly()), fill=255)
    d.ellipse([(hx - hr) * SS, (hy - hr) * SS, (hx + hr) * SS, (hy + hr) * SS], fill=0)
pal_mask = mask_from(pal_draw)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def affine_mask(m, scale, alpha):
    if abs(scale - 1) < 1e-4:
        return m * alpha
    im = Image.fromarray((m * 255).astype(np.uint8))
    cx, cy = OX, OY
    inv = 1 / scale
    im = im.transform((W, H), Image.AFFINE, (inv, 0, cx - cx * inv, 0, inv, cy - cy * inv), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255 * alpha


def splat(acc_c, acc_a, s, t):
    p = clamp((t - s["t0"]) / s["dur"])
    if p <= 0:
        return
    e = ease_out_cubic(p)
    sc = 0.3 + 0.7 * ease_out_back(p, 1.3)
    an = 1.0
    op = ease_out_cubic(clamp(p * 2.0))
    cx, cy = s["cx"], s["cy"]
    rot = math.radians(s["rot"])
    rx, ry = s["rx"] * K * sc * an, s["ry"] * K * sc / an
    px, py = to_px(cx, cy)
    R = max(rx, ry) + 2
    x0, x1 = int(max(0, px - R)), int(min(W, px + R + 1))
    y0, y1 = int(max(0, py - R)), int(min(H, py + R + 1))
    if x0 >= x1 or y0 >= y1:
        return
    X = xx[y0:y1, x0:x1] - px
    Y = yy[y0:y1, x0:x1] - py
    c, sn = math.cos(rot), math.sin(rot)
    u = (X * c + Y * sn) / rx
    v = (-X * sn + Y * c) / ry
    r = np.sqrt(u * u + v * v)
    offs, ops = zip(*s["prof"])
    a = np.interp(r, offs, ops).astype(np.float32) * op
    a[r > 1] = 0
    a = a[..., None]
    acc_c[y0:y1, x0:x1] = s["col"] * a + acc_c[y0:y1, x0:x1] * (1 - a)
    acc_a[y0:y1, x0:x1] = a + acc_a[y0:y1, x0:x1] * (1 - a)


# ---- wordmark ------------------------------------------------------------
# "Splatelier" fades in under the palette while the brush leaves, then holds.
from PIL import ImageFont
TITLE = "Splatelier"
TITLE_SIZE = int(0.150 * W)        # em size in px
TITLE_TRACK = 0.025 * TITLE_SIZE    # letter spacing
TITLE_CY = int(0.768 * H)          # vertical centre of the text
T_TITLE = T_EXIT - 0.45            # start of the fade-in
D_TITLE = 0.55                     # fade-in length


def build_title_mask():
    font = ImageFont.truetype(args.font, TITLE_SIZE)
    adv = [font.getlength(ch) for ch in TITLE]
    total = sum(adv) + TITLE_TRACK * (len(TITLE) - 1)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    x = (W - total) / 2
    for ch, a in zip(TITLE, adv):
        d.text((x, TITLE_CY), ch, font=font, fill=255, anchor="ls")
        x += a + TITLE_TRACK
    return np.asarray(im, np.float32) / 255

TITLE_MASK = None if args.no_title else build_title_mask()


def draw_title(img, t):
    if TITLE_MASK is None or t <= T_TITLE:
        return img
    e = ease_out_cubic(clamp((t - T_TITLE) / D_TITLE))
    shift = int(round((1 - e) * 0.014 * H))          # rises into place
    m = np.roll(TITLE_MASK, shift, axis=0)[..., None] * e
    return IVORY * m + img * (1 - m)


def frame(t):
    img = np.empty((H, W, 3), np.float32); img[:] = BG
    # palette: settles in over 0.0-0.45s
    pp = clamp(t / 0.45)
    pa = ease_out_cubic(pp)
    psc = 0.94 + 0.06 * ease_out_back(pp, 1.6) if pp > 0 else 0.94
    sh = affine_mask(shadow_mask, psc, 0.3 * pa)[..., None]
    img = img * (1 - sh)
    pm = affine_mask(pal_mask, psc, pa)[..., None]
    img = IVORY * pm + img * (1 - pm)
    # multiply group
    gc = np.zeros((H, W, 3), np.float32); ga = np.zeros((H, W, 1), np.float32)
    for s in S:
        if s["layer"] == "m":
            splat(gc, ga, s, t)
    img = img * (1 - ga + gc)           # multiply with opaque backdrop
    # normal group
    nc = np.zeros((H, W, 3), np.float32); na = np.zeros((H, W, 1), np.float32)
    for s in S:
        if s["layer"] == "n":
            splat(nc, na, s, t)
    img = nc + img * (1 - na)
    img = draw_title(np.clip(img, 0, 1), t)
    img = draw_brush(np.clip(img, 0, 1), t)
    return np.clip(img, 0, 1)


# ---- the fude (Japanese brush) ------------------------------------------
from PIL import ImageFilter

def bez(p0, p1, p2, p3, n=24):
    return cubic(np.array(p0, float), np.array(p1, float), np.array(p2, float), np.array(p3, float), n).tolist()

HAIR = (bez((0, 0), (40, -10), (90, -24), (130, -22)) + bez((130, -22), (150, -21), (156, -12), (156, 0))
        + bez((156, 0), (156, 12), (150, 21), (130, 22)) + bez((130, 22), (90, 24), (40, 10), (0, 0)))

def rect(x0, x1, h0, h1=None):
    h1 = h0 if h1 is None else h1
    return [(x0, -h0), (x1, -h1), (x1, h1), (x0, h0)]

def clip_x(poly, L):
    return [(x, y) for x, y in poly if x <= L] + [(L, 0)]

C_HAIR, C_COLLAR = (239, 230, 214), (43, 31, 27)
C_BAMBOO, C_HI, C_NODE = (216, 184, 120), (236, 211, 158), (168, 133, 78)
BK = 0.95 * K

def contact(t):
    return float(np.mean([brush_at(t + d)[2] for d in np.linspace(-0.035, 0.035, 9)]))

def paint_colour(t):
    done = [s for s in S if s["t0"] <= t]
    if not done:
        return PURPLE
    return max(done, key=lambda s: s["t0"])["col"]

def draw_brush(img, t):
    bx, by, _ = brush_at(t)
    bx2, _, _ = brush_at(t + 1 / 120)
    vx = (bx2 - bx) * 120
    c = contact(t)
    lift = 1 - c
    th = math.radians(-42 - max(-12, min(12, vx * 0.006)))
    sc = BK * (1 + 0.05 * lift)
    tx, ty = to_px(bx, by)
    ct, st = math.cos(th), math.sin(th)
    def T(poly, ox=0.0, oy=0.0):
        return [(tx + ox + sc * (x * ct - y * st), ty + oy + sc * (x * st + y * ct)) for x, y in poly]
    pc = tuple(int(v * 255) for v in paint_colour(t))
    shapes = [
        (rect(178, 650, 16, 15), C_BAMBOO, 255),
        (rect(178, 650, -4, -4) [:2] + rect(178, 650, 10, 9)[:2][::-1], C_HI, 255),
        (rect(328, 334, 17), C_NODE, 255), (rect(488, 494, 16), C_NODE, 255),
        (rect(626, 652, 16, 14), C_COLLAR, 255),
        (rect(148, 180, 25, 22), C_COLLAR, 255),
        (HAIR, C_HAIR, 255),
        (clip_x(HAIR, 100), pc, 110), (clip_x(HAIR, 72), pc, 170), (clip_x(HAIR, 44), pc, 240),
    ]
    silhouette = [rect(148, 652, 25, 16), HAIR]
    off = (10 + 20 * lift) * K, (12 + 26 * lift) * K
    allpts = np.array([p for poly, _, _ in shapes for p in T(poly)] + [p for poly in silhouette for p in T(poly, *off)])
    x0, y0 = np.floor(allpts.min(0) - 30).astype(int)
    x1, y1 = np.ceil(allpts.max(0) + 30).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x0 >= x1 or y0 >= y1:
        return img
    bw, bh, Q = x1 - x0, y1 - y0, 3
    loc = lambda poly, ox=0, oy=0: [((x - x0) * Q, (y - y0) * Q) for x, y in T(poly, ox, oy)]
    # shadow
    shm = Image.new("L", (bw * Q, bh * Q), 0); d = ImageDraw.Draw(shm)
    for poly in silhouette:
        d.polygon(loc(poly, *off), fill=255)
    shm = shm.resize((bw, bh), Image.LANCZOS).filter(ImageFilter.GaussianBlur((5 + 9 * lift) * K))
    sa = np.asarray(shm, np.float32)[..., None] / 255 * 0.34
    reg = img[y0:y1, x0:x1]
    reg = reg * (1 - sa)
    # brush body
    layer = Image.new("RGBA", (bw * Q, bh * Q), (0, 0, 0, 0))
    for poly, col, a in shapes:
        ov = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).polygon(loc(poly), fill=col + (a,))
        layer = Image.alpha_composite(layer, ov)
    layer = np.asarray(layer.resize((bw, bh), Image.LANCZOS), np.float32) / 255
    la = layer[..., 3:4]
    reg = layer[..., :3] * la + reg * (1 - la)
    img[y0:y1, x0:x1] = reg
    return img


N = int(round(args.dur * args.slow * args.fps))
frames = range(N) if args.only < 0 else [args.only]
for i in frames:
    t = i / args.fps / args.slow
    out = (frame(t) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(args.out, f"f{i:04d}.png"))
print("frames:", len(frames))
