#!/usr/bin/env python3
"""Independent check of a denoise output DNG against its source.

Deliberately not Swift and not the tool's own reader: a bug in `SourceDNGDecode` that both sides
share would cancel out of a Swift-only comparison. This decodes the SOF3 tiles itself and prints
every geometry invariant next to the two numbers it compared, so the report can quote them.

    python3 dng-invariants.py SRC.dng OUT.dng [--flat x,y,size] [--phases [--phase-tol 0.02]]

Exit status 0 means every invariant matched.

`--phases` additionally compares the mean level of each of the four CFA phases. That is the only
check that catches a *shifted mosaic*: the CFA tag is copied verbatim either way, so a build that
hard-coded `rggb` writes a file whose metadata says `bggr` and whose picture is a checkerboard.
"""
import struct
import sys
import array

COMP = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


class DNG:
    def __init__(self, path):
        self.path = path
        self.data = open(path, 'rb').read()
        d = self.data
        if d[0:2] != b'II' or struct.unpack('<H', d[2:4])[0] != 42:
            raise SystemExit(f'{path}: not a little-endian TIFF')
        self.ifd0 = self._ifd(struct.unpack('<I', d[4:8])[0])
        raw = self.ifd0
        if 330 in self.ifd0:
            off = struct.unpack('<I', self.ifd0[330][2][0:4])[0]
            cand = self._ifd(off)
            if cand.get(262, (0, 0, b''))[2][:2] == struct.pack('<H', 32803):
                raw = cand
        elif self.ifd0.get(262, (0, 0, b''))[2][:2] != struct.pack('<H', 32803):
            raise SystemExit(f'{path}: no CFA IFD')
        self.raw = raw

    def _ifd(self, off):
        d = self.data
        n = struct.unpack('<H', d[off:off + 2])[0]
        out = {}
        for i in range(n):
            b = off + 2 + 12 * i
            tag, typ, cnt = struct.unpack('<HHI', d[b:b + 8])
            sz = COMP.get(typ, 1) * cnt
            val = d[b + 8:b + 8 + sz] if sz <= 4 else d[struct.unpack('<I', d[b + 8:b + 12])[0]:][:sz]
            out[tag] = (typ, cnt, val)
        return out

    def num(self, tag, where=None):
        e = (where or self.raw).get(tag)
        if e is None:
            return []
        typ, cnt, val = e
        if typ == 1:
            return list(val[:cnt])
        if typ == 3:
            return list(struct.unpack(f'<{cnt}H', val[:2 * cnt]))
        if typ == 4:
            return list(struct.unpack(f'<{cnt}I', val[:4 * cnt]))
        if typ in (5, 10):
            out = []
            for i in range(cnt):
                a, b = struct.unpack('<II', val[i * 8:i * 8 + 8])
                out.append(a / b if b else 0.0)
            return out
        return []

    def scalar(self, tag, where=None, default=None):
        v = self.num(tag, where)
        return v[0] if v else default

    # --- mosaic decoding (compression 1 and 7) ---
    def mosaic(self):
        w = self.scalar(256)
        h = self.scalar(257)
        comp = self.scalar(259)
        samples = array.array('H', bytes(2 * w * h))
        if 322 in self.raw and 324 in self.raw:
            tw = self.scalar(322)
            th = self.scalar(323)
            offs = self.num(324)
            cnts = self.num(325)
            nx = (w + tw - 1) // tw
            ny = (h + th - 1) // th
            idx = 0
            for ty in range(ny):
                for tx in range(nx):
                    blob = self.data[offs[idx]:offs[idx] + cnts[idx]]
                    if comp == 1:
                        px = array.array('H')
                        px.frombytes(blob[:2 * tw * th])
                    elif comp == 7:
                        px = decode_sof3(blob, tw, th)
                    else:
                        raise SystemExit(f'unsupported compression {comp}')
                    for y in range(th):
                        dy = ty * th + y
                        if dy >= h:
                            break
                        cw = min(tw, w - tx * tw)
                        base = dy * w + tx * tw
                        samples[base:base + cw] = px[y * tw:y * tw + cw]
                    idx += 1
            return samples, w, h
        if 273 in self.raw:
            if comp != 1:
                raise SystemExit(f'strip + compression {comp} unsupported')
            offs = self.num(273)
            cnts = self.num(279)
            cur = 0
            for o, c in zip(offs, cnts):
                rows = c // 2 // w
                for r in range(rows):
                    if cur >= h:
                        break
                    base = o + r * w * 2
                    samples[cur * w:(cur + 1) * w] = array.array('H', self.data[base:base + 2 * w])
                    cur += 1
            return samples, w, h
        raise SystemExit('no tile or strip geometry')


def decode_sof3(data, expect_w, expect_h):
    """Lossless-JPEG (SOF3) decoder, written from the spec rather than mirrored from the encoder."""
    i = 2
    precision, comps, cw, h, pred = 16, 1, expect_w, expect_h, 1
    tables, sos = [], 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xD8:
            i += 2
            continue
        if i + 4 > len(data):
            break
        ln = (data[i + 2] << 8) | data[i + 3]
        body = i + 4
        if m == 0xC3:
            precision = data[body]
            h = (data[body + 1] << 8) | data[body + 2]
            cw = (data[body + 3] << 8) | data[body + 4]
            comps = data[body + 5]
        elif m == 0xC4:
            p = body
            while p < i + 2 + ln:
                is_dc = (data[p] & 0xF0) == 0
                tid = data[p] & 0x0F
                counts = list(data[p + 1:p + 17])
                total = sum(counts)
                syms = list(data[p + 17:p + 17 + total])
                p += 17 + total
                if not is_dc:
                    continue
                code, si, entries = 0, 0, {}
                for l in range(1, 17):
                    for _ in range(counts[l - 1]):
                        entries[(l, code)] = syms[si]
                        si += 1
                        code += 1
                    code <<= 1
                while len(tables) <= tid:
                    tables.append({})
                tables[tid] = entries
        elif m == 0xDA:
            p = body
            ns = data[p]
            p += 1
            sel = []
            for _ in range(ns):
                p += 1
                sel.append(data[p] >> 4)
                p += 1
            pred = data[p]
            sos = i + 2 + ln
            tables = [(tables[s] if s < len(tables) else {}) for s in sel] if sel else tables
        else:
            pass
        if m == 0xDA:
            break
        i += 2 + ln

    acc, nbits, pos = 0, 0, sos
    mask = (1 << precision) - 1
    half = 1 << (precision - 1)

    def bit():
        nonlocal acc, nbits, pos
        while nbits == 0:
            b = data[pos]
            pos += 1
            if b == 0xFF:
                pos += 1
            acc, nbits = b, 8
        nbits -= 1
        return (acc >> nbits) & 1

    out = array.array('H', bytes(2 * cw * comps * h))
    for y in range(h):
        row = y * comps * cw
        for x in range(cw):
            for c in range(comps):
                tbl = tables[c] if c < len(tables) else {}
                cat, code = 0, 0
                found = False
                for l in range(1, 17):
                    code = (code << 1) | bit()
                    if (l, code) in tbl:
                        cat = tbl[(l, code)]
                        found = True
                        break
                if not found:
                    raise SystemExit('corrupt SOF3 stream')
                d = 0
                if cat > 0:
                    v = 0
                    for _ in range(cat):
                        v = (v << 1) | bit()
                    d = v if v >= (1 << (cat - 1)) else v - (1 << cat) + 1
                here = row + x * comps + c
                if pred == 1:
                    if x > 0:
                        pv = out[here - comps]
                    elif y > 0:
                        pv = out[here - comps * cw]
                    else:
                        pv = half
                else:
                    pv = out[here - comps * cw] if y > 0 else half
                out[here] = (pv + d) & mask
    return out


def adjacent_variance(samples, w, x0, y0, size):
    """Mean of squared horizontal first differences inside one square — a stand-in for noise."""
    tot, n = 0.0, 0
    for y in range(y0, y0 + size):
        base = y * w + x0
        for x in range(base, base + size - 1):
            d = float(samples[x + 1]) - float(samples[x])
            tot += d * d
            n += 1
    return tot / max(n, 1)


def phase_means(samples, w, h):
    """Mean of each of the four CFA phases, in mosaic row-major order."""
    totals = [0.0] * 4
    counts = [0] * 4
    for y in range(h):
        base = y * w
        for x in range(w):
            p = (y % 2) * 2 + (x % 2)
            totals[p] += samples[base + x]
            counts[p] += 1
    return [totals[i] / max(counts[i], 1) for i in range(4)]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = DNG(sys.argv[1]), DNG(sys.argv[2])
    flat = None
    if '--flat' in sys.argv:
        v = sys.argv[sys.argv.index('--flat') + 1].split(',')
        flat = (int(v[0]), int(v[1]), int(v[2]))
    want_phases = '--phases' in sys.argv
    phase_tol = 0.02
    if '--phase-tol' in sys.argv:
        phase_tol = float(sys.argv[sys.argv.index('--phase-tol') + 1])

    print(f'source : {sys.argv[1]}')
    print(f'output : {sys.argv[2]}')
    ok = True

    def check(name, a, b):
        nonlocal ok
        good = a == b
        ok = ok and good
        print(f'  {"PASS" if good else "FAIL"}  {name:<16} {a}  vs  {b}')

    check('width', src.scalar(256), out.scalar(256))
    check('height', src.scalar(257), out.scalar(257))
    # Compression is not compared with the source: the writer always emits 7, and the fixture is
    # uncompressed (1). The invariant is "the output is lossless JPEG", plus photometric stays CFA.
    check('photometric', src.scalar(262), out.scalar(262))
    check('bitsPerSample', src.scalar(258), out.scalar(258))
    check('samplesPerPixel', src.scalar(277), out.scalar(277))
    check('blackRepeatDim', src.num(50713), out.num(50713))
    check('blackLevel', src.num(50714), out.num(50714))
    check('whiteLevel', src.num(50717), out.num(50717))
    check('cfaPatternDim', src.num(33421), out.num(33421))
    check('cfaPattern', src.num(33422), out.num(33422))
    check('activeArea', src.num(50829), out.num(50829))
    check('defaultCropSize', src.num(50720), out.num(50720))
    check('compressionIs7', 7, out.scalar(259))
    check('photometricIsCFA', 32803, out.scalar(262))
    print(f'  info  source compression={src.scalar(259)} → output compression={out.scalar(259)}, '
          f'photometric={out.scalar(262)}')

    # Previews survive too: a viewer should not have to regenerate a thumbnail.
    print(f'  info  preview in source={273 in src.ifd0} in output={273 in out.ifd0}')

    if want_phases:
        # The CFA *tag* is copied verbatim, so a hard-coded phase in the reader leaves it correct
        # and the picture wrong. This is the only check that sees it: put a distinct level on each
        # phase in the source and require the output to keep the same phase→level mapping.
        a, w, h = src.mosaic()
        b, w2, h2 = out.mosaic()
        assert (w, h) == (w2, h2)
        pa, pb = phase_means(a, w, h), phase_means(b, w, h)
        print('\nCFA phase means (mosaic row-major: 00, 01, 10, 11)')
        print('  source ' + '  '.join(f'{v:8.1f}' for v in pa))
        print('  output ' + '  '.join(f'{v:8.1f}' for v in pb))
        # Only meaningful when the source phases are actually distinguishable.
        spread = max(pa) - min(pa)
        if spread < 50:
            print('  info  the source phases are not distinguishable (spread '
                  f'{spread:.1f}); phase order not asserted')
        else:
            order_a = sorted(range(4), key=lambda i: pa[i])
            order_b = sorted(range(4), key=lambda i: pb[i])
            same = order_a == order_b
            worst = max(abs(pb[i] - pa[i]) / max(abs(pa[i]), 1) for i in range(4))
            print(f'  {"PASS" if same else "FAIL"}  phase order {order_a} vs {order_b} '
                  f'(worst relative shift {100 * worst:.1f} %)')
            ok = ok and same and worst <= phase_tol
        # A shift by one pixel swaps the phase assignment; that is the failure this catches.
        if len(set(round(v) for v in pa)) == 4:
            print('  info  four distinct source levels — a 1 px shift is detectable here')

    if flat:
        a, w, h = src.mosaic()
        b, w2, h2 = out.mosaic()
        assert (w, h) == (w2, h2)
        x0, y0, size = flat
        va = adjacent_variance(a, w, x0, y0, size)
        vb = adjacent_variance(b, w, x0, y0, size)
        mn_a = sum(a[y * w + x] for y in range(y0, y0 + size) for x in range(x0, x0 + size)) / (size * size)
        mn_b = sum(b[y * w + x] for y in range(y0, y0 + size) for x in range(x0, x0 + size)) / (size * size)
        ratio = vb / va if va else float('inf')
        print(f'\nflat region {size}² at ({x0},{y0})')
        print(f'  adjacent-difference variance: before {va:.2f}  after {vb:.2f}  ratio {ratio:.4f}')
        print(f'  mean:                         before {mn_a:.2f}  after {mn_b:.2f}  '
              f'change {100 * (mn_b - mn_a) / mn_a:+.2f} %')
        gate = ratio < 0.5 and abs(mn_b - mn_a) <= 0.05 * mn_a
        print(f'  {"PASS" if gate else "FAIL"}  ratio < 0.5 and |mean change| <= 5 %')
        ok = ok and gate

    print('\nRESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
