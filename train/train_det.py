#!/usr/bin/env python3
"""Small-object detector training: whole frame in, object centres out.

The pipeline before this relied on a fitted lattice whose constants were measured
on one device.  This model learns where the objects are from the picture itself,
so the spacing, the vertical offset and the count no longer have to be guessed.

Target format (data/boxes.jsonl):
    {"image": "...", "size": [w, h], "boxes": [[x, y, w, h, label], ...]}

Usage:
    python3 train/train_det.py --epochs 60 --out runs/det/v1
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import ImageEnhance
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 384, 192   # fixed canvas; the source frames keep a ~2.1 aspect
CROP_TOP = 1.0    # keep only the top fraction of the frame (play band)
DIM_P = float(os.environ.get('DET_DIM', '0.35'))
SCALE_JITTER = 0.0     # 随机缩放幅度，0 关闭（默认与原行为一致）
STRIDE = 2


class DetNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        b1, b2, b3 = base, base * 2, base * 4
        self.e1 = blk(3, b1)
        self.e2 = blk(b1, b2)
        self.e3 = blk(b2, b3)
        self.up2 = nn.ConvTranspose2d(b3, b2, 2, 2)
        self.d2 = blk(b2 * 2, b2)
        self.hm = nn.Conv2d(b2, 1, 1)
        self.wh = nn.Conv2d(b2, 2, 1)
        self.hm.bias.data.fill_(-4.0)
        # a zeroed size head sits exactly on the ReLU knee and never recovers
        self.wh.bias.data.fill_(8.0)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        y = self.d2(torch.cat([self.up2(x3), x2], 1))
        return self.hm(y), F.relu(self.wh(y))


def load_records(path):
    recs = []
    for line in open(path):
        r = json.loads(line)
        if r['boxes']:
            recs.append(r)
    return recs


def make_target(size, boxes, nw, nh):
    hm = np.zeros((1, nh // STRIDE, nw // STRIDE), np.float32)
    wh = np.zeros((2, nh // STRIDE, nw // STRIDE), np.float32)
    mask = np.zeros((1, nh // STRIDE, nw // STRIDE), np.float32)
    sx, sy = nw / size[0], nh / size[1]
    for x, y, w, h, *_ in boxes:
        cx = (x + w / 2) * sx / STRIDE
        cy = (y + h / 2) * sy / STRIDE
        bw, bh = w * sx / STRIDE, h * sy / STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < hm.shape[2] and 0 <= iy < hm.shape[1]):
            continue
        r = max(1.0, min(bw, bh) / 5.0)
        yy, xx = np.mgrid[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        hm[0, yy, xx] = np.maximum(hm[0, yy, xx], g)
        wh[0, iy, ix] = bw
        wh[1, iy, ix] = bh
        mask[0, iy, ix] = 1.0
    return hm, wh, mask


def focal(pred, gt):
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = (gt > 0.9).float()
    neg = 1 - pos
    pos_loss = -pos * torch.log(p) * (1 - p) ** 2
    neg_loss = -neg * torch.log(1 - p) * p ** 2 * (gt < 0.9).float()
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def find_frame(name):
    for d in (os.path.join(ROOT, 'data/train_frames'), os.path.join(ROOT, 'data/train_frames_auto'),
              os.path.join(ROOT, 'data/test_frames')):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def build_batch(recs, frames_dir=None):
    xs, hms, whs, ms = [], [], [], []
    ims, tg = [], []
    for r in recs:
        fp = r.get('frame')
        im = Image.open(fp if fp and os.path.exists(fp) else find_frame(r['image'])).convert('RGB')
        # some objects sit under a translucent overlay (measured: 0.87 of the
        # normal brightness); without this the detector never fires there
        if DIM_P and random.random() < DIM_P:
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.82, 0.92))
        boxes, size = r['boxes'], list(im.size)
        if CROP_TOP < 1.0:
            ch = int(size[1] * CROP_TOP)
            im = im.crop((0, 0, size[0], ch))
            boxes = [b for b in boxes if b[1] + b[3] <= ch]
            size = [size[0], ch]
        nw, nh = W, H
        im = im.resize((nw, nh), Image.LANCZOS)
        k = 1.0
        if SCALE_JITTER > 0:
            k = random.uniform(1.0 - SCALE_JITTER, 1.0)
        if k < 0.999:
            sx, sy = nw / size[0], nh / size[1]
            boxes = [[b[0] * sx * k + (nw - nw * k) / 2,
                      b[1] * sy * k + (nh - nh * k) / 2,
                      b[2] * sx * k, b[3] * sy * k, *b[4:]] for b in boxes]
            small = im.resize((max(8, int(nw * k)), max(8, int(nh * k))), Image.LANCZOS)
            canvas = Image.new('RGB', (nw, nh), (0, 0, 0))
            canvas.paste(small, (int((nw - small.width) / 2), int((nh - small.height) / 2)))
            im = canvas
            size = [nw, nh]
        hm, wh, m = make_target(size, boxes, nw, nh)
        xs.append(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
        hms.append(hm[0]); whs.append(wh); ms.append(m[0])
    xs = torch.from_numpy(np.stack(xs))
    hms = torch.from_numpy(np.stack(hms)).unsqueeze(1)
    whs = torch.from_numpy(np.stack(whs))
    ms = torch.from_numpy(np.stack(ms)).unsqueeze(1)
    return xs, hms, whs, ms


def decode(hm, wh):
    hm = torch.sigmoid(hm)[0, 0]
    pool = F.max_pool2d(hm[None, None], 5, 1, 2)
    keep = (pool[0, 0] == hm) & (hm > 0.35)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    out = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        out.append((x, y, float(hm[y, x]), float(wh[0, 0, y, x]), float(wh[0, 1, y, x])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boxes', default=os.path.join(ROOT, 'data/train_boxes_all.jsonl'))
    ap.add_argument('--frames', default=os.path.join(ROOT, 'data/train_frames'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs/det/v1'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=int, default=0, help='input width, 0 keeps the default')
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--crop-top', type=float, default=1.0)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--scale-jitter', type=float, default=0.0, help='随机缩放幅度，0 关闭')
    args = ap.parse_args()

    global W, H, CROP_TOP, SCALE_JITTER
    SCALE_JITTER = args.scale_jitter
    if args.width:
        W = args.width
    if args.height:
        H = args.height
    CROP_TOP = args.crop_top
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    recs = load_records(args.boxes)
    random.shuffle(recs)
    split = max(1, int(len(recs) * 0.85))
    tr, va = recs[:split], recs[split:]
    os.makedirs(args.out, exist_ok=True)

    model = DetNet(args.base)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    print(f'训练 {len(tr)} 张，验证 {len(va)} 张，参数 {sum(p.numel() for p in model.parameters()):,}')

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        random.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), args.bs):
            x, hm, wh, m = build_batch(tr[i:i + args.bs], args.frames)
            ph, pwh = model(x)
            loss = focal(ph, hm) * 4 + (F.l1_loss(pwh * m, wh * m, reduction='sum')
                                        / m.sum().clamp(min=1)) / 20
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(x)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for i in range(0, len(va), args.bs):
                x, hm, wh, m = build_batch(va[i:i + args.bs], args.frames)
                ph, pwh = model(x)
                vl += float(focal(ph, hm) * 4 + (F.l1_loss(pwh * m, wh * m, reduction='sum')
                                                 / m.sum().clamp(min=1)) / 20) * len(x)
        vl /= max(len(va), 1)
        print(f'epoch {ep + 1:3d}  训练 {tot / max(len(tr), 1):.4f}  验证 {vl:.4f}')
        if vl < best:
            best = vl
            torch.save({'model': model.state_dict(), 'w': W, 'h': H, 'stride': STRIDE,
                        'crop_top': CROP_TOP, 'base': args.base}, args.out + '/best.pt')
    print('best', round(best, 4), '->', args.out + '/best.pt')


if __name__ == '__main__':
    main()
