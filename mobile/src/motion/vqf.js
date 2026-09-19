// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
// SPDX-License-Identifier: MIT
// JavaScript port of the 6D path of BasicVQF 2.1.2 (not the full bias estimator).
// https://github.com/dlaidig/vqf/tree/v2.1.2 — see VQF-LICENSE.txt.

export const norm = (v) => Math.hypot(...v);
export const normalize = (v) => {
  const n = norm(v);
  return n < Number.EPSILON ? [...v] : v.map((x) => x / n);
};
export const conjugate = ([w, x, y, z]) => [w, -x, -y, -z];
export function multiply([w, x, y, z], [a, b, c, d]) {
  return [w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
    w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a];
}
export function rotate([w, x, y, z], [a, b, c]) {
  return [
    (1-2*y*y-2*z*z)*a + 2*(y*x-w*z)*b + 2*(w*y+z*x)*c,
    2*(w*z+y*x)*a + (1-2*x*x-2*z*z)*b + 2*(y*z-x*w)*c,
    2*(z*x-w*y)*a + 2*(w*x+z*y)*b + (1-2*x*x-2*y*y)*c,
  ];
}

// Shortest rotation taking a unit gravity vector to +Z. Yaw is unobservable.
export function inclinationQuaternion(gravity) {
  const [x, y, z] = normalize(gravity);
  const w = Math.sqrt(Math.max(0, Math.min(1, (z + 1) / 2)));
  return w > 1e-6 ? [w, 0.5*y/w, -0.5*x/w, 0] : [0, 1, 0, 0];
}

export class BasicVQF {
  constructor(period = 0.01, tauAcc = 3) {
    if (!Number.isFinite(period) || period <= 0 || !Number.isFinite(tauAcc) || tauAcc <= 0) {
      throw new Error('VQF requires positive finite sampling and filter periods.');
    }
    this.period = period;
    this.tauAcc = tauAcc;
    const c = Math.tan(Math.SQRT2 * period / (2 * tauAcc));
    const d = c*c + Math.SQRT2*c + 1;
    this.b = tauAcc < period/2 ? [1, 0, 0] : [c*c/d, 2*c*c/d, c*c/d];
    this.a = tauAcc < period/2 ? [0, 0] : [2*(c*c-1)/d, (1-Math.SQRT2*c+c*c)/d];
    this.reset();
  }

  reset() {
    this.gyrQuat = [1, 0, 0, 0];
    this.accQuat = [1, 0, 0, 0];
    this.count = 0;
    this.sum = [0, 0, 0];
    this.state = null;
  }

  filter(acc) {
    const [b0, b1, b2] = this.b;
    const [a1, a2] = this.a;
    if (!this.state) {
      this.count++;
      this.sum = this.sum.map((s, i) => s + acc[i]);
      const out = this.sum.map((s) => s / this.count);
      if (this.count * this.period >= this.tauAcc) {
        this.state = out.map((v) => [v*(1-b0), v*(b2-a2)]);
      }
      return out;
    }
    return acc.map((v, i) => {
      const s = this.state[i];
      const y = b0*v + s[0];
      s[0] = b1*v - a1*y + s[1];
      s[1] = b2*v - a2*y;
      return y;
    });
  }

  update(gyr, acc) {
    if (gyr.length !== 3 || acc.length !== 3 || ![...gyr, ...acc].every(Number.isFinite)) {
      throw new Error('VQF expects finite three-axis SI samples.');
    }
    const n = norm(gyr);
    if (n > Number.EPSILON) {
      const angle = n * this.period;
      const s = Math.sin(angle/2) / n;
      this.gyrQuat = normalize(multiply(this.gyrQuat, [Math.cos(angle/2), ...gyr.map((v) => s*v)]));
    }
    if (norm(acc) > 0) {
      const filtered = this.filter(rotate(this.gyrQuat, acc));
      const gravity = rotate(this.accQuat, filtered);
      this.accQuat = normalize(multiply(inclinationQuaternion(gravity), this.accQuat));
    }
    return multiply(this.accQuat, this.gyrQuat);
  }
}
