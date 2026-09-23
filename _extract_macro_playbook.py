"""One-off: extract JS playbook literals from macro_simulator.html → JSON."""
from __future__ import annotations
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "macro_simulator.html")
OUT = os.path.join(HERE, "macro_sim_playbook.json")


class JSLiteral:
    def parse(self, src: str):
        self.s = src
        self.n = len(src)
        self.i = 0
        self._skip()
        val = self._value()
        return val

    def _peek(self):
        return self.s[self.i] if self.i < self.n else ""

    def _skip(self):
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r\n":
                self.i += 1
                continue
            if self.s.startswith("//", self.i):
                self.i = self.s.find("\n", self.i)
                if self.i < 0:
                    self.i = self.n
                continue
            if self.s.startswith("/*", self.i):
                end = self.s.find("*/", self.i + 2)
                self.i = self.n if end < 0 else end + 2
                continue
            break

    def _value(self):
        self._skip()
        c = self._peek()
        if c == "{":
            return self._object()
        if c == "[":
            return self._array()
        if c in "'\"":
            return self._string()
        if self.s.startswith("null", self.i):
            self.i += 4
            return None
        if self.s.startswith("true", self.i):
            self.i += 4
            return True
        if self.s.startswith("false", self.i):
            self.i += 5
            return False
        if c == "-" or c.isdigit() or (c == "." and self.i + 1 < self.n and self.s[self.i + 1].isdigit()):
            return self._number()
        raise ValueError(f"bad value at {self.i}: {self.s[self.i:self.i+40]!r}")

    def _object(self):
        self.i += 1
        out = {}
        while True:
            self._skip()
            if self._peek() == "}":
                self.i += 1
                return out
            key = self._key()
            self._skip()
            if self._peek() != ":":
                raise ValueError(f"expected : after {key} at {self.i}")
            self.i += 1
            out[key] = self._value()
            self._skip()
            if self._peek() == ",":
                self.i += 1
                continue
            if self._peek() == "}":
                self.i += 1
                return out
            raise ValueError(f"expected , or }} at {self.i}: {self.s[self.i:self.i+40]!r}")

    def _key(self):
        self._skip()
        c = self._peek()
        if c in "'\"":
            return self._string()
        start = self.i
        if not (c.isalpha() or c in "_$"):
            raise ValueError(f"bad key at {self.i}: {self.s[self.i:self.i+40]!r}")
        self.i += 1
        while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] in "_$"):
            self.i += 1
        return self.s[start:self.i]

    def _array(self):
        self.i += 1
        out = []
        while True:
            self._skip()
            if self._peek() == "]":
                self.i += 1
                return out
            out.append(self._value())
            self._skip()
            if self._peek() == ",":
                self.i += 1
                continue
            if self._peek() == "]":
                self.i += 1
                return out
            raise ValueError(f"expected , or ] at {self.i}: {self.s[self.i:self.i+40]!r}")

    def _string(self):
        q = self.s[self.i]
        self.i += 1
        chars = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == "\\":
                self.i += 1
                e = self.s[self.i]
                chars.append({
                    "n": "\n", "r": "\r", "t": "\t", "\\": "\\",
                    "'": "'", '"': '"', "/": "/",
                }.get(e, e))
                self.i += 1
                continue
            if c == q:
                self.i += 1
                return "".join(chars)
            chars.append(c)
            self.i += 1
        raise ValueError("unterminated string")

    def _number(self):
        start = self.i
        if self._peek() == "-":
            self.i += 1
        while self.i < self.n and (self.s[self.i].isdigit() or self.s[self.i] in ".eE+-"):
            self.i += 1
        raw = self.s[start:self.i]
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)


def extract_const(html: str, name: str) -> str:
    m = re.search(rf"const {name}\s*=", html)
    if not m:
        raise SystemExit(f"const {name} not found")
    return html[m.end():]


def brace_slice(src: str, opener="{") -> str:
    closer = "}" if opener == "{" else "]"
    i = 0
    while i < len(src) and src[i] in " \t\r\n":
        i += 1
    if src[i] != opener:
        raise SystemExit(f"expected {opener}, got {src[i:i+20]!r}")
    depth = 0
    in_str = None
    esc = False
    j = i
    while j < len(src):
        c = src[j]
        if in_str:
            if esc:
                esc = False
                j += 1
                continue
            if c == "\\":
                esc = True
                j += 1
                continue
            if c == in_str:
                in_str = None
            j += 1
            continue
        if src.startswith("//", j):
            nl = src.find("\n", j)
            j = len(src) if nl < 0 else nl + 1
            continue
        if src.startswith("/*", j):
            end = src.find("*/", j + 2)
            j = len(src) if end < 0 else end + 2
            continue
        if c in "'\"":
            in_str = c
            j += 1
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
        j += 1
    raise SystemExit("unbalanced")


def main():
    html = open(HTML, encoding="utf-8").read()
    p = JSLiteral()
    db = p.parse(brace_slice(extract_const(html, "DB")))
    scens = p.parse(brace_slice(extract_const(html, "SCENS")))
    narrs = p.parse(brace_slice(extract_const(html, "NARRS")))
    angles = p.parse(brace_slice(extract_const(html, "ANGLES"), opener="["))
    base_px = p.parse(brace_slice(extract_const(html, "BASE_PX")))
    payload = dict(db=db, scens=scens, narrs=narrs, angles=angles, base_px=base_px)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    n = sum(len(v) for v in db.values() if isinstance(v, list))
    print(f"wrote {OUT}  groups={list(db)}  names={n}  scens={list(scens)}")


if __name__ == "__main__":
    main()
