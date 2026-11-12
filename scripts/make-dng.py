#!/usr/bin/env python3
"""Make a test DNG: re-lay-out an existing CFA DNG with a new size, CFA phase or flat field.

    python3 scripts/make-dng.py SRC.dng OUT.dng [--crop WxH] [--cfa rggb|bggr|grbg|gbrg]
                                             [--flat LEVEL] [--noise SIGMA] [--seed N]

Writes an **uncompressed** (Compression=1) CFA DNG whose IFD0 carries the pixel-domain, colour and
geometry tags copied from the source. EXIF/GPS/preview are dropped — this is a geometry fixture, and
whatever it does not copy it must not silently pretend to copy.

Why not the writer: this has to produce files *before* the tool has run, at sizes and phases the
shipped fixtures do not cover. Building the TIFF here also keeps the check independent — if the
tool's own writer had a layout bug, a fixture made by it would be wrong in the same way.
"""
import struct
import sys
import math
import random
import array

COMP = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
LAYOUT = {254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325,
          330, 339, 34665, 34853}
CFA_LETTER = {'r': 0, 'g': 1, 'b': 2}


def u16(v):
    return struct.pack('<H', v)


def u32(v):
    return struct.pack('<I', v)


def read_ifd(data, off):
    n = struct.unpack('<H', data[off:off + 2])[0]
    out = {}
    for i in range(n):
        b = off + 2 + 12 * i
        tag, typ, cnt = struct.unpack('<HHI', data[b:b + 8])
        size = COMP.get(typ, 1) * cnt
        if size <= 4:
            val = data[b + 8:b + 8 + size]
        else:
            p = struct.unpack('<I', data[b + 8:b + 12])[0]
            val = data[p:p + size]
        out[tag] = (typ, cnt, val)
    return out


def decode_raw(path):
    data = open(path, 'rb').read()
    ifd0 = read_ifd(data, struct.unpack('<I', data[4:8])[0])
    raw = ifd0
    if 330 in ifd0:
        cand = read_ifd(data, struct.unpack('<I', ifd0[330][2][:4])[0])
        if cand.get(262, (0, 0, b''))[2][:2] == struct.pack('<H', 32803):
            raw = cand
    if raw.get(262, (0, 0, b''))[2][:2] != struct.pack('<H', 32803):
        raise SystemExit(f'{path}: no CFA image')
    w = struct.unpack('<I', raw[256][2][:4] if raw[256][0] == 4 else raw[256][2].ljust(4, b'\0'))[0]
    w = raw[256][2]
    w = struct.unpack('<H', w[:2])[0] if raw[256][0] == 3 else struct.unpack('<I', w[:4])[0]
    h = raw[257][2]
    h = struct.unpack('<H', h[:2])[0] if raw[257][0] == 3 else struct.unpack('<I', h[:4])[0]
    comp = struct.unpack('<H', raw[259][2][:2])[0]
    if comp != 1:
        raise SystemExit(f'{path}: only uncompressed sources are supported (got {comp})')
    off = struct.unpack('<I', raw[273][2][:4])[0]
    n = w * h
    samples = array.array('H')
    samples.frombytes(data[off:off + 2 * n])
    return ifd0, raw, samples, w, h


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]

    def opt(name, default=None):
        if name in args:
            return args[args.index(name) + 1]
        return default

    ifd0, raw, samples, w, h = decode_raw(src)

    crop = opt('--crop')
    if crop:
        nw, nh = (int(v) for v in crop.lower().split('x'))
    else:
        nw, nh = w, h

    cfa = opt('--cfa')
    flat = opt('--flat')
    noise = float(opt('--noise', '0'))
    seed = int(opt('--seed', '12345'))
    phase_gains = opt('--phase-gains')          # e.g. "1,1.1,1.2,1.3" for phases 00,01,10,11

    if flat is not None or noise or phase_gains:
        level = float(flat) if flat is not None else 2000.0
        rng = random.Random(seed)
        gains = [float(v) for v in phase_gains.split(',')] if phase_gains else [1, 1, 1, 1]
        if len(gains) != 4:
            raise SystemExit('--phase-gains wants 4 comma-separated multipliers')
        samples = array.array('H', bytes(2 * nw * nh))
        for y in range(nh):
            for x in range(nw):
                g = gains[(y % 2) * 2 + (x % 2)]
                v = level * g + (rng.gauss(0, noise) if noise else 0)
                samples[y * nw + x] = max(0, min(65535, int(round(v))))
    else:
        if (nw, nh) != (w, h):
            out = array.array('H', bytes(2 * nw * nh))
            for y in range(nh):
                out[y * nw:(y + 1) * nw] = samples[y * w:y * w + nw]
            samples = out

    # Tags carried over verbatim, minus everything that describes the old layout.
    preserved = {t: e for t, e in ifd0.items() if t not in LAYOUT}
    for t, e in raw.items():
        if t not in LAYOUT:
            preserved[t] = e

    def put(tag, typ, cnt, payload):
        preserved[tag] = (typ, cnt, payload)

    put(254, 4, 1, u32(0))
    put(256, 4, 1, u32(nw))
    put(257, 4, 1, u32(nh))
    put(258, 3, 1, u16(16))
    put(259, 3, 1, u16(1))
    put(262, 3, 1, u16(32803))
    put(277, 3, 1, u16(1))
    put(278, 4, 1, u32(nh))
    put(284, 3, 1, u16(1))
    put(339, 3, 1, u16(1))

    if cfa:
        put(33421, 3, 2, u16(2) + u16(2))
        put(33422, 1, 4, bytes(CFA_LETTER[c] for c in cfa.lower()))

    # ActiveArea / DefaultCropSize must describe the new extent, or the file lies about its picture.
    # Encode with the source's own type width: this fixture stores ActiveArea as SHORT[4], and
    # writing LONGs behind a SHORT count makes every offset-shifted value read as 0.
    def encode(typ, values):
        if typ == 3:
            return b''.join(u16(v) for v in values)
        if typ == 4:
            return b''.join(u32(v) for v in values)
        raise SystemExit(f'unsupported tag type {typ}')

    if 50829 in preserved:
        typ = preserved[50829][0]
        put(50829, typ, 4, encode(typ, (0, 0, nh, nw)))
    if 50720 in preserved:
        typ = preserved[50720][0]
        put(50720, typ, 2, encode(typ, (nw, nh)))

    entries = sorted(preserved.items())
    head = 2 + 12 * len(entries) + 4
    payloads = []
    payload_size = 0
    for tag, (typ, cnt, val) in entries:
        if len(val) > 4:
            payloads.append((tag, val))
            payload_size += len(val) + (len(val) % 2)

    data_start = 8 + head + payload_size
    data_start += (4 - data_start % 4) % 4

    put(273, 4, 1, u32(data_start))
    put(279, 4, 1, u32(2 * nw * nh))
    entries = sorted(preserved.items())

    out = bytearray()
    out += b'II' + u16(42) + u32(8)
    out += u16(len(entries))
    payload = bytearray()
    payload_base = 8 + 2 + 12 * len(entries) + 4
    for tag, (typ, cnt, val) in entries:
        out += u16(tag) + u16(typ) + u32(cnt)
        if len(val) <= 4:
            out += val.ljust(4, b'\0')
        else:
            out += u32(payload_base + len(payload))
            payload += val
            if len(payload) % 2:
                payload += b'\0'
    out += u32(0)
    out += payload
    while len(out) < data_start:
        out += b'\0'
    out += samples.tobytes()

    open(dst, 'wb').write(out)
    print(f'{dst}: {nw}×{nh} CFA={cfa or "unchanged"} {len(out)} bytes, '
          f'{len(entries)} tags, strip at {data_start}')


if __name__ == '__main__':
    main()
