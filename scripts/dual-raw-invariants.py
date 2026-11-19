#!/usr/bin/env python3
"""Check a `--dual-raw` DNG: preview in IFD0, two CFA raws in `SubIFDs`, original first.

    python3 scripts/dual-raw-invariants.py SRC.dng DUAL.dng

Independent of the Swift code on purpose (same reasoning as `dng-invariants.py`): a decoder bug
shared by the writer and the reader would cancel out of a Swift-only comparison. The SOF3 decoder
is imported from `dng-invariants.py` rather than re-written here.

What it asserts, and why each one matters:

  1. IFD0 is the preview page: `NewSubfileType = 1`, strips (`273`/`279`), not CFA. That is the
     only layout in which ImageIO resolves the *original* — it scans `SubIFDs` and takes the first
     subfiletype-0 raw, so a file with the original in IFD0 would hand the denoised layer to the
     system decoder.
  2. `330 SubIFDs` has exactly two entries, both CFA / compression 7 / full size / subfiletype 0.
  3. `SubIFDs[0]` decodes **pixel for pixel** to the source's raw mosaic (max |diff| == 0), and
     `SubIFDs[1]` decodes to something different (RMS reported). If the order were reversed, a
     standard reader would show the denoised layer as the picture.
  4. The alternate carries `270 ImageDescription` and no metadata copies; EXIF/GPS pointers stay on
     IFD0.
  5. IFD0's next-IFD pointer is 0 (the preview is not a page chain head any more).

Exit status 0 means every check passed.
"""
import importlib.util
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_spec = importlib.util.spec_from_file_location(
    "dng_invariants", os.path.join(_HERE, "dng-invariants.py"))
_dng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dng)
decode_sof3 = _dng.decode_sof3

COMP = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
CFA = 32803


class Doc:
    def __init__(self, path):
        self.path = path
        self.data = open(path, 'rb').read()
        if self.data[0:2] != b'II' or struct.unpack('<H', self.data[2:4])[0] != 42:
            raise SystemExit(f'{path}: not a little-endian TIFF')
        self.ifd0_offset = struct.unpack('<I', self.data[4:8])[0]
        self.ifd0 = self.read_ifd(self.ifd0_offset)

    def read_ifd(self, off):
        d = self.data
        n = struct.unpack('<H', d[off:off + 2])[0]
        out = {}
        for i in range(n):
            b = off + 2 + 12 * i
            tag, typ, cnt = struct.unpack('<HHI', d[b:b + 8])
            size = COMP.get(typ, 1) * cnt
            if size <= 4:
                val = d[b + 8:b + 8 + size]
            else:
                start = struct.unpack('<I', d[b + 8:b + 12])[0]
                val = d[start:start + size]
            out[tag] = (typ, cnt, val)
        return out

    def next_ifd(self, off):
        n = struct.unpack('<H', self.data[off:off + 2])[0]
        return struct.unpack('<I', self.data[off + 2 + 12 * n:off + 2 + 12 * n + 4])[0]

    def nums(self, ifd, tag):
        e = ifd.get(tag)
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

    def scalar(self, ifd, tag, default=None):
        v = self.nums(ifd, tag)
        return v[0] if v else default

    def text(self, ifd, tag):
        e = ifd.get(tag)
        return None if e is None else e[2].rstrip(b'\x00').decode('latin-1')

    # --- mosaic of one specific IFD (tiles or strips, compression 1 or 7) ---
    def mosaic(self, ifd):
        w = int(self.scalar(ifd, 256))
        h = int(self.scalar(ifd, 257))
        comp = int(self.scalar(ifd, 259))
        out = [0] * (w * h)
        if 322 in ifd and 324 in ifd:
            tw = int(self.scalar(ifd, 322))
            th = int(self.scalar(ifd, 323))
            offs = self.nums(ifd, 324)
            cnts = self.nums(ifd, 325)
            nx, ny = (w + tw - 1) // tw, (h + th - 1) // th
            if len(offs) != nx * ny or len(cnts) != len(offs):
                raise SystemExit(f'{self.path}: tile table has {len(offs)} offsets for {nx * ny} tiles')
            idx = 0
            for ty in range(ny):
                for tx in range(nx):
                    blob = self.data[offs[idx]:offs[idx] + cnts[idx]]
                    if len(blob) != cnts[idx]:
                        raise SystemExit(f'{self.path}: tile {idx} runs past EOF')
                    if comp == 1:
                        px = list(struct.unpack(f'<{tw * th}H', blob[:2 * tw * th]))
                    elif comp == 7:
                        px = decode_sof3(blob, tw, th)
                    else:
                        raise SystemExit(f'unsupported compression {comp}')
                    for y in range(th):
                        dy = ty * th + y
                        if dy >= h:
                            break
                        cw = min(tw, w - tx * tw)
                        out[dy * w + tx * tw:dy * w + tx * tw + cw] = px[y * tw:y * tw + cw]
                    idx += 1
            return out, w, h
        if 273 in ifd:
            if comp != 1:
                raise SystemExit(f'strip + compression {comp} unsupported')
            cur = 0
            for o, c in zip(self.nums(ifd, 273), self.nums(ifd, 279)):
                rows = c // 2 // w
                for r in range(rows):
                    if cur >= h:
                        break
                    base = o + r * w * 2
                    out[cur * w:(cur + 1) * w] = struct.unpack(f'<{w}H', self.data[base:base + 2 * w])
                    cur += 1
            return out, w, h
        raise SystemExit(f'{self.path}: no tile or strip geometry in this IFD')


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dual = Doc(sys.argv[1]), Doc(sys.argv[2])
    ok = True

    def check(name, good, detail=''):
        nonlocal ok
        ok = ok and bool(good)
        print(f'  {"PASS" if good else "FAIL"}  {name:<34} {detail}')

    print(f'source : {sys.argv[1]}')
    print(f'dual   : {sys.argv[2]} ({len(dual.data)} bytes)')

    # --- 1. IFD0 is the preview page ---------------------------------------------------------
    print('\nIFD0 (preview)')
    print(f'  offset {dual.ifd0_offset}, {len(dual.ifd0)} entries')
    print('  ' + '  '.join(str(t) for t in sorted(dual.ifd0)))
    check('IFD0 is NOT a raw (262)', dual.scalar(dual.ifd0, 262) != CFA,
          f'262={dual.scalar(dual.ifd0, 262)}')
    check('IFD0 subfiletype is 1', dual.scalar(dual.ifd0, 254) == 1,
          f'254={dual.scalar(dual.ifd0, 254)}')
    check('IFD0 has strips (273/279)', 273 in dual.ifd0 and 279 in dual.ifd0,
          f'{len(dual.nums(dual.ifd0, 273))} strip(s), '
          f'{dual.scalar(dual.ifd0, 256)}x{dual.scalar(dual.ifd0, 257)}')
    check('IFD0.next == 0', dual.next_ifd(dual.ifd0_offset) == 0,
          f'next={dual.next_ifd(dual.ifd0_offset)}')
    check('EXIF/GPS hang off IFD0', 34665 in dual.ifd0 and 34853 in dual.ifd0,
          f'exif={dual.scalar(dual.ifd0, 34665)} gps={dual.scalar(dual.ifd0, 34853)}')

    # --- 2. SubIFDs: two CFA raws, both full size, subfiletype 0 ------------------------------
    print('\nIFD0 330 SubIFDs')
    sub = dual.nums(dual.ifd0, 330)
    check('SubIFDs has exactly 2 entries', len(sub) == 2, f'{sub}')
    entries = []
    for i, off in enumerate(sub):
        ifd = dual.read_ifd(off)
        entries.append(ifd)
        print(f'  [{i}] @{off}: ' + '  '.join(str(t) for t in sorted(ifd)))
        check(f'SubIFDs[{i}] is CFA (262)', dual.scalar(ifd, 262) == CFA,
              f'262={dual.scalar(ifd, 262)}')
        check(f'SubIFDs[{i}] compression 7', dual.scalar(ifd, 259) == 7,
              f'259={dual.scalar(ifd, 259)}')
        check(f'SubIFDs[{i}] subfiletype 0', dual.scalar(ifd, 254) == 0,
              f'254={dual.scalar(ifd, 254)}')
        check(f'SubIFDs[{i}] size matches', (dual.scalar(ifd, 256), dual.scalar(ifd, 257)) ==
              (dual.scalar(entries[0], 256), dual.scalar(entries[0], 257)),
              f'{dual.scalar(ifd, 256)}x{dual.scalar(ifd, 257)}')

    # --- 3. pixels: [0] is the source, [1] is the denoised -------------------------------------------------
    print('\npixels')
    a, w, h = src.mosaic(src.ifd0)          # the fixture keeps its raw in IFD0
    if len(entries) == 2:
        b, w2, h2 = dual.mosaic(entries[0])
        check('original geometry matches source', (w, h) == (w2, h2), f'{w}x{h} vs {w2}x{h2}')
        maxdiff = max(abs(int(x) - int(y)) for x, y in zip(a, b))
        exact = sum(1 for x, y in zip(a, b) if x == y)
        check('SubIFDs[0] == source, max |diff| 0', maxdiff == 0,
              f'max|diff|={maxdiff}, identical {exact}/{len(a)}')

        c, w3, h3 = dual.mosaic(entries[1])
        check('alternate geometry matches', (w3, h3) == (w2, h2), f'{w3}x{h3}')
        n = min(len(a), len(c))
        sq = sum((int(c[i]) - int(a[i])) ** 2 for i in range(n))
        rms = (sq / n) ** 0.5
        diff = [abs(int(c[i]) - int(a[i])) for i in range(n)]
        changed = sum(1 for d in diff if d > 0)
        print(f'  alternate vs source: rms={rms:.3f}  max|diff|={max(diff)}  '
              f'changed={changed}/{n} ({100 * changed / n:.1f} %)')
        check('SubIFDs[1] differs from source', rms > 0 and max(diff) > 0, f'rms {rms:.4f}')
        same = sum(1 for x, y in zip(b, c) if x == y)
        check('the two layers are distinguishable', same < len(b), f'identical {same}/{len(b)}')

        # --- 4. the alternate is a self-describing raw, minus develop-look data --------------
        alt = entries[1]
        check('alternate carries 270 description', 270 in alt, repr(dual.text(alt, 270)))
        # Those describe how to *develop* the file, and a second copy makes ImageIO apply the
        # profile/opcodes twice. EXIF/GPS/XMP have one owner: IFD0.
        forbidden = sorted(set(alt) & {330, 34665, 34853, 700, 50740, 50721, 50722, 50728,
                                       50938, 50939, 50981, 50982, 51022, 51041})
        check('alternate has no develop-look data', not forbidden, f'forbidden={forbidden}')
        needed = [33421, 33422, 50713, 50714, 50717, 50719, 50720, 50829]
        missing = [t for t in needed if t not in alt]
        check('alternate is self-describing', not missing, f'missing={missing}')
        check('original keeps the metadata', 50829 in entries[0] and 33422 in entries[0]
              and 51022 in entries[0],
              'ActiveArea/CFAPattern/OpcodeList2 present')
        # The tile payloads must not be shared — a pointer bug would satisfy every tag check.
        check('layers have disjoint tile payloads',
              set(dual.nums(entries[0], 324)).isdisjoint(set(dual.nums(alt, 324))),
              f'{len(dual.nums(entries[0], 324))} + {len(dual.nums(alt, 324))} tiles')

    print('\nRESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
