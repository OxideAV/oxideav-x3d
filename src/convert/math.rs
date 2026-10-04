//! Small vector / quaternion / matrix helpers (f32, row-major 4×4
//! matrices acting on column vectors — the mesh3d convention).

/// 4×4 matrix, `m[row][col]`.
pub type Mat4 = [[f32; 4]; 4];

/// Identity matrix.
pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn len(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Normalise, returning `None` for (near-)zero or non-finite vectors.
pub fn normalize(a: [f32; 3]) -> Option<[f32; 3]> {
    let l = len(a);
    if l.is_finite() && l > 1e-12 {
        Some(scale(a, 1.0 / l))
    } else {
        None
    }
}

/// Quaternion (xyzw) from an X3D axis-angle rotation `[x, y, z, angle]`.
pub fn quat_from_axis_angle(r: [f32; 4]) -> [f32; 4] {
    let Some(axis) = normalize([r[0], r[1], r[2]]) else {
        return [0.0, 0.0, 0.0, 1.0];
    };
    let a = r[3];
    if !a.is_finite() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let (s, c) = (a * 0.5).sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

/// Axis-angle `[x, y, z, angle]` from a quaternion (xyzw).
pub fn axis_angle_from_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !(n.is_finite() && n > 0.0) {
        return [0.0, 0.0, 1.0, 0.0];
    }
    let mut q = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    if q[3] < 0.0 {
        q = [-q[0], -q[1], -q[2], -q[3]];
    }
    let w = q[3].clamp(-1.0, 1.0);
    let angle = 2.0 * w.acos();
    match normalize([q[0], q[1], q[2]]) {
        Some(a) if angle.abs() > 1e-7 => [a[0], a[1], a[2], angle],
        _ => [0.0, 0.0, 1.0, 0.0],
    }
}

pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Rotate vector `v` by unit quaternion `q`.
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let p = [v[0], v[1], v[2], 0.0];
    let qc = [-q[0], -q[1], -q[2], q[3]];
    let r = quat_mul(quat_mul(q, p), qc);
    [r[0], r[1], r[2]]
}

/// Shortest-arc rotation (quaternion) taking unit `from` onto unit `to`.
pub fn quat_between(from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
    let (Some(f), Some(t)) = (normalize(from), normalize(to)) else {
        return [0.0, 0.0, 0.0, 1.0];
    };
    let d = dot(f, t);
    if d > 1.0 - 1e-7 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d < -1.0 + 1e-7 {
        // 180°: any axis perpendicular to `f`.
        let axis = normalize(cross(f, [1.0, 0.0, 0.0]))
            .or_else(|| normalize(cross(f, [0.0, 1.0, 0.0])))
            .unwrap_or([0.0, 0.0, 1.0]);
        return [axis[0], axis[1], axis[2], 0.0];
    }
    let c = cross(f, t);
    let w = 1.0 + d;
    let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + w * w).sqrt();
    [c[0] / n, c[1] / n, c[2] / n, w / n]
}

/// Rotation matrix of a unit quaternion.
pub fn mat_from_quat(q: [f32; 4]) -> Mat4 {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            0.0,
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            0.0,
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat_translate(t: [f32; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[0][3] = t[0];
    m[1][3] = t[1];
    m[2][3] = t[2];
    m
}

pub fn mat_scale(s: [f32; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[0][0] = s[0];
    m[1][1] = s[1];
    m[2][2] = s[2];
    m
}

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0.0f32; 4]; 4];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

/// General 4×4 inverse (cofactor expansion); `None` when singular.
pub fn mat_inverse(m: &Mat4) -> Option<Mat4> {
    let a: Vec<f64> = m.iter().flatten().map(|&v| v as f64).collect();
    let mut inv = [0.0f64; 16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
        + a[9] * a[7] * a[14]
        + a[13] * a[6] * a[11]
        - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
        - a[8] * a[7] * a[14]
        - a[12] * a[6] * a[11]
        + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
        + a[8] * a[7] * a[13]
        + a[12] * a[5] * a[11]
        - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
        - a[8] * a[6] * a[13]
        - a[12] * a[5] * a[10]
        + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
        - a[9] * a[3] * a[14]
        - a[13] * a[2] * a[11]
        + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
        + a[8] * a[3] * a[14]
        + a[12] * a[2] * a[11]
        - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
        - a[8] * a[3] * a[13]
        - a[12] * a[1] * a[11]
        + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
        + a[8] * a[2] * a[13]
        + a[12] * a[1] * a[10]
        - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15]
        + a[5] * a[3] * a[14]
        + a[13] * a[2] * a[7]
        - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
        - a[4] * a[3] * a[14]
        - a[12] * a[2] * a[7]
        + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15]
        + a[4] * a[3] * a[13]
        + a[12] * a[1] * a[7]
        - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
        - a[4] * a[2] * a[13]
        - a[12] * a[1] * a[6]
        + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11]
        - a[5] * a[3] * a[10]
        - a[9] * a[2] * a[7]
        + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11]
        + a[4] * a[3] * a[10]
        + a[8] * a[2] * a[7]
        - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11]
        - a[4] * a[3] * a[9]
        - a[8] * a[1] * a[7]
        + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10]
        + a[4] * a[2] * a[9]
        + a[8] * a[1] * a[6]
        - a[8] * a[2] * a[5];
    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if !det.is_finite() || det.abs() < 1e-20 {
        return None;
    }
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..16 {
        r[i / 4][i % 4] = (inv[i] / det) as f32;
    }
    Some(r)
}

/// X3D `Transform` matrix (ISO/IEC 19775-1 10.4.4):
/// `P' = T × C × R × SR × S × -SR × -C × P`.
pub fn x3d_transform_matrix(
    translation: [f32; 3],
    rotation: [f32; 4],
    scale_v: [f32; 3],
    scale_orientation: [f32; 4],
    center: [f32; 3],
) -> Mat4 {
    let t = mat_translate(translation);
    let c = mat_translate(center);
    let r = mat_from_quat(quat_from_axis_angle(rotation));
    let sq = quat_from_axis_angle(scale_orientation);
    let sr = mat_from_quat(sq);
    let sr_inv = mat_from_quat([-sq[0], -sq[1], -sq[2], sq[3]]);
    let s = mat_scale(scale_v);
    let nc = mat_translate([-center[0], -center[1], -center[2]]);
    let mut m = mat_mul(&t, &c);
    m = mat_mul(&m, &r);
    m = mat_mul(&m, &sr);
    m = mat_mul(&m, &s);
    m = mat_mul(&m, &sr_inv);
    mat_mul(&m, &nc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quat_roundtrip() {
        let r = [0.0, 1.0, 0.0, 1.0];
        let q = quat_from_axis_angle(r);
        let back = axis_angle_from_quat(q);
        assert!((back[1] - 1.0).abs() < 1e-5 && (back[3] - 1.0).abs() < 1e-5);
        let v = quat_rotate(
            quat_from_axis_angle([0.0, 1.0, 0.0, std::f32::consts::FRAC_PI_2]),
            [0.0, 0.0, -1.0],
        );
        assert!((v[0] + 1.0).abs() < 1e-5, "{v:?}");
        let q = quat_between([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]);
        let v = quat_rotate(q, [0.0, 0.0, -1.0]);
        assert!((v[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inverse() {
        let m = x3d_transform_matrix(
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 0.0, 0.7],
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 0.3],
            [0.5, 0.0, 0.0],
        );
        let i = mat_inverse(&m).unwrap();
        let p = mat_mul(&m, &i);
        for (r, row) in p.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-4);
            }
        }
    }
}
