"""
ankle_kinematics.py

Per-side wrapper around the corrected 2-motor ankle IK/FK (axis fixes +
roll/pitch swap confirmed against hardware earlier in this project):
  - motor rotation axis = X (disc lies in the Y-Z plane)
  - roll(phi)  -> rotation about Y
  - pitch(theta) -> rotation about X

The measured geometry constants below (A1, A2, O, C1_init, C2_init) were
taken from one physical ankle (the one kinematics.py/kinematics_fixed.py was
built from). Which leg that is (left or right) has NOT been confirmed here.

*** TODO: confirm before use ***
  1. Which leg (left/right) do these measured points actually belong to?
  2. The other leg's version below is a MIRROR (x -> -x) of these points,
     assuming perfect bilateral symmetry. This is an assumption, not a
     measurement -- verify against the real robot before trusting it.
  3. Confirm the real CAN motor IDs for both ankle actuators on both legs.
     (kinematics.py's own comments call them "motor7"/"motor8", which does
     NOT match the placeholder mapping in robot_constant.py where 7/8 are
     the RIGHT leg's hipz/hipx. This mismatch must be resolved before
     wiring real CAN ids in single_leg_controller.py.)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Software test envelope requested for the ankle (on top of whatever the
# linkage can kinematically reach -- see FEASIBLE checks below).
ROLL_LIMIT_DEG: Tuple[float, float] = (-25.0, 25.0)
PITCH_LIMIT_DEG: Tuple[float, float] = (-40.0, 40.0)


class AnkleUnreachableError(ValueError):
    """Raised when a requested (pitch, roll) has no valid motor solution."""


class AnkleKinematics:
    def __init__(self, side: str = "right"):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        mirror = -1.0 if side == "left" else 1.0  # TODO: verify bilateral symmetry assumption

        # Measured geometry (see module docstring: confirmed for ONE leg only).
        self.A1 = np.array([-26.10, 74.00, -255.00]) * np.array([mirror, 1, 1])
        self.A2 = np.array([-80.90, 74.00, -313.00]) * np.array([mirror, 1, 1])
        self.O = np.array([-53.50, 74.00, -393.00]) * np.array([mirror, 1, 1])
        C1_init = np.array([-8.10, 111.343, -398.00]) * np.array([mirror, 1, 1])
        C2_init = np.array([-98.90, 111.343, -398.00]) * np.array([mirror, 1, 1])

        self.R = 40.0
        self.t = 18.0
        self.OFFSET_SIGN_1 = 1.0
        self.OFFSET_SIGN_2 = -1.0
        # Both A1/A2 wired/driven so that +alpha = clockwise for both motors
        # (same sign, unlike the earlier +1/-1 placeholder). Confirmed
        # convention, not a TODO anymore -- if this is ever found to be
        # wrong on the other leg / other hardware rev, update here.
        self.ROTATION_SIGN_1 = 1.0
        self.ROTATION_SIGN_2 = 1.0

        self.ALPHA1_START = 0.0
        self.ALPHA2_START = 0.0

        a1m = self.ROTATION_SIGN_1 * self.ALPHA1_START
        a2m = self.ROTATION_SIGN_2 * self.ALPHA2_START
        self.B1_init = np.array([
            self.A1[0] + self.OFFSET_SIGN_1 * self.t,
            self.A1[1] + self.R * np.cos(a1m),
            self.A1[2] + self.R * np.sin(a1m),
        ])
        self.B2_init = np.array([
            self.A2[0] + self.OFFSET_SIGN_2 * self.t,
            self.A2[1] + self.R * np.cos(a2m),
            self.A2[2] + self.R * np.sin(a2m),
        ])

        self.L1 = float(np.linalg.norm(self.B1_init - C1_init))
        self.L2 = float(np.linalg.norm(self.B2_init - C2_init))
        self.C1_local = C1_init - self.O
        self.C2_local = C2_init - self.O

    def _solve_motor_angle(self, D, R, L, target_angle, offset_sign, rotation_sign):
        A_trig = 2 * R * D[1]
        B_trig = 2 * R * D[2]
        D_sq = (D[0] - offset_sign * self.t) ** 2 + D[1] ** 2 + D[2] ** 2
        K = D_sq + R ** 2 - L ** 2
        rho = np.sqrt(A_trig ** 2 + B_trig ** 2)
        ratio = K / rho
        feasible = abs(ratio) <= 1.0
        psi = np.arctan2(A_trig, B_trig)
        asin_val = np.arcsin(np.clip(ratio, -1.0, 1.0))
        cand1 = -psi + asin_val
        cand2 = -psi + np.pi - asin_val
        target_math = rotation_sign * target_angle

        def wrap_near(a, tgt):
            return a + 2 * np.pi * np.round((tgt - a) / (2 * np.pi))

        cand1 = wrap_near(cand1, target_math)
        cand2 = wrap_near(cand2, target_math)
        chosen = cand1 if abs(cand1 - target_math) <= abs(cand2 - target_math) else cand2
        return rotation_sign * chosen, feasible, ratio

    def solve_ik(self, pitch_deg: float, roll_deg: float, *, enforce_test_envelope: bool = True):
        """
        Returns (alpha1_deg, alpha2_deg). Raises AnkleUnreachableError if the
        target has no valid motor solution (instead of silently clipping),
        or if enforce_test_envelope=True and the target falls outside
        ROLL_LIMIT_DEG / PITCH_LIMIT_DEG.
        """
        if enforce_test_envelope:
            if not (ROLL_LIMIT_DEG[0] <= roll_deg <= ROLL_LIMIT_DEG[1]):
                raise AnkleUnreachableError(
                    f"roll={roll_deg:.2f} outside test envelope {ROLL_LIMIT_DEG}"
                )
            if not (PITCH_LIMIT_DEG[0] <= pitch_deg <= PITCH_LIMIT_DEG[1]):
                raise AnkleUnreachableError(
                    f"pitch={pitch_deg:.2f} outside test envelope {PITCH_LIMIT_DEG}"
                )

        phi, theta = np.radians(roll_deg), np.radians(pitch_deg)
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)

        R_roll = np.array([[c_p, 0.0, s_p], [0.0, 1.0, 0.0], [-s_p, 0.0, c_p]])
        C1p = self.O + R_roll @ self.C1_local
        C2p = self.O + R_roll @ self.C2_local
        R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, c_t, -s_t], [0.0, s_t, c_t]])
        C1f = self.O + R_pitch @ (C1p - self.O)
        C2f = self.O + R_pitch @ (C2p - self.O)

        a1, feasible1, _ = self._solve_motor_angle(
            C1f - self.A1, self.R, self.L1, self.ALPHA1_START,
            self.OFFSET_SIGN_1, self.ROTATION_SIGN_1,
        )
        a2, feasible2, _ = self._solve_motor_angle(
            C2f - self.A2, self.R, self.L2, self.ALPHA2_START,
            self.OFFSET_SIGN_2, self.ROTATION_SIGN_2,
        )
        if not (feasible1 and feasible2):
            raise AnkleUnreachableError(
                f"pitch={pitch_deg:.2f}, roll={roll_deg:.2f} has no valid rod solution "
                f"(motor1 feasible={feasible1}, motor2 feasible={feasible2})"
            )

        # Both motor angles operate around 0 deg, so wrap both symmetrically
        # to (-180, 180] deg. (Previously a1 was pushed into [0, 360), which
        # put a wrap discontinuity right next to the operating region -- bad
        # for the finite-difference Jacobian used by the torque-control
        # methods below.)
        a1 = (a1 + np.pi) % (2 * np.pi) - np.pi
        a2 = (a2 + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(a1), np.degrees(a2)

    def solve_fk(self, alpha1_deg: float, alpha2_deg: float, *, guess_pitch_deg=0.0,
                 guess_roll_deg=0.0, max_iter=120, tol=1e-7):
        a1, a2 = np.radians(alpha1_deg), np.radians(alpha2_deg)
        a1m = self.ROTATION_SIGN_1 * a1
        a2m = self.ROTATION_SIGN_2 * a2
        B1 = np.array([self.A1[0] + self.OFFSET_SIGN_1 * self.t,
                       self.A1[1] + self.R * np.cos(a1m), self.A1[2] + self.R * np.sin(a1m)])
        B2 = np.array([self.A2[0] + self.OFFSET_SIGN_2 * self.t,
                       self.A2[1] + self.R * np.cos(a2m), self.A2[2] + self.R * np.sin(a2m)])

        phi, theta = np.radians(guess_roll_deg), np.radians(guess_pitch_deg)
        eps = 1e-6
        for _ in range(max_iter):
            def geom(p, t):
                c_p, s_p = np.cos(p), np.sin(p)
                c_t, s_t = np.cos(t), np.sin(t)
                R_roll = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
                R_pitch = np.array([[1, 0, 0], [0, c_t, -s_t], [0, s_t, c_t]])
                C1p = self.O + R_roll @ self.C1_local
                C2p = self.O + R_roll @ self.C2_local
                return self.O + R_pitch @ (C1p - self.O), self.O + R_pitch @ (C2p - self.O)

            C1, C2 = geom(phi, theta)
            f1 = np.sum((C1 - B1) ** 2) - self.L1 ** 2
            f2 = np.sum((C2 - B2) ** 2) - self.L2 ** 2
            F = np.array([f1, f2])
            if np.linalg.norm(F) < tol:
                break
            C1p_, C2p_ = geom(phi + eps, theta)
            df1_dphi = (np.sum((C1p_ - B1) ** 2) - self.L1 ** 2 - f1) / eps
            df2_dphi = (np.sum((C2p_ - B2) ** 2) - self.L2 ** 2 - f2) / eps
            C1t_, C2t_ = geom(phi, theta + eps)
            df1_dtheta = (np.sum((C1t_ - B1) ** 2) - self.L1 ** 2 - f1) / eps
            df2_dtheta = (np.sum((C2t_ - B2) ** 2) - self.L2 ** 2 - f2) / eps
            J = np.array([[df1_dphi, df1_dtheta], [df2_dphi, df2_dtheta]])
            delta = np.linalg.solve(J + np.eye(2) * 1e-11, -F)
            phi += delta[0]
            theta += delta[1]

        return np.degrees(theta), np.degrees(phi)  # (pitch, roll)

    def is_feasible(self, pitch_deg: float, roll_deg: float) -> bool:
        try:
            self.solve_ik(pitch_deg, roll_deg, enforce_test_envelope=False)
            return True
        except AnkleUnreachableError:
            return False

    # ------------------------------------------------------------------
    # Torque-based (MIT-style) control support.
    #
    # Position control on this linkage doesn't produce the torque you
    # actually want at the ankle, so instead we:
    #   1) treat (pitch, roll) as a virtual joint and run MIT-style
    #      PD(+feedforward) on it to get a desired (tau_pitch, tau_roll)
    #   2) map that virtual torque pair to real motor torques (tau1, tau2)
    #      for A1/A2 using the torque-transpose of the position Jacobian
    #      J = d(alpha)/d(pitch,roll), linearized at the CURRENT measured
    #      pose (not the desired one, since that's the real geometry the
    #      linkage has right now):
    #
    #           tau_alpha^T * dalpha = tau_pr^T * dpr        (virtual work,
    #                                                          lossless
    #                                                          transmission)
    #           dalpha = J * dpr
    #        => tau_alpha = (J^T)^{-1} @ tau_pr
    #
    #   3) send tau1/tau2 to the A/B motors' own MIT-mode command as pure
    #      feedforward torque with kp=kd=0 on the motor side (position
    #      control stays off; the motors just source/sink the requested
    #      torque).
    #
    # NOTE ON UNITS: these three methods are radian-based (tau in Nm,
    # kp in Nm/rad, kd in Nm/(rad/s)) since that matches the standard MIT
    # actuator convention. The rest of this file (solve_ik/solve_fk/
    # ROLL_LIMIT_DEG/...) is degree-based -- convert at the call site with
    # np.radians()/np.degrees() as needed. Don't mix degrees into the
    # torque math below, or kp/kd will silently be off by a factor of
    # 180/pi.
    # ------------------------------------------------------------------

    def jacobian_pr_to_motor_analytic(self, pitch_rad: float, roll_rad: float) -> np.ndarray:
        """
        Closed-form Jacobian J = d(alpha1, alpha2) / d(pitch, roll) [rad/rad],
        exact (no finite differences, no eps, no wrap/unwrap concerns).

        Derivation: alpha_i comes from solving the rod-length constraint
            |B_i(beta_i) - C_i|^2 = L_i^2,   beta_i = rotation_sign_i * alpha_i
        for beta_i (this is exactly what _solve_motor_angle does in closed
        form via the A_trig*cos+B_trig*sin = K identity). Implicitly
        differentiating that constraint w.r.t. D_i = C_i - A_i gives

            d(beta_i)/d(D_i) = (C_i - B_i) / (R * (D_i[2]*cos(beta_i) - D_i[1]*sin(beta_i)))

        i.e. the rod vector (C_i - B_i, magnitude L_i) divided by a scalar
        that is exactly zero at this rod's own kinematic singularity (rod
        tangent to the crank circle -- the same condition as
        _solve_motor_angle's feasibility boundary |ratio| -> 1). Then
        chain-rule through C_i(pitch,theta) = O + R_pitch @ R_roll @ C_i_local
        (using the analytic derivatives of R_roll, R_pitch) to get
        d(beta_i)/d(pitch), d(beta_i)/d(roll), and finally
        d(alpha_i)/d(*) = rotation_sign_i * d(beta_i)/d(*).

        Raises AnkleUnreachableError if either rod is at/past its own
        feasibility boundary or within 1e-9 of its singular denominator.
        """
        phi, theta = roll_rad, pitch_rad  # phi=roll (Y-axis), theta=pitch (X-axis)
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)

        R_roll = np.array([[c_p, 0.0, s_p], [0.0, 1.0, 0.0], [-s_p, 0.0, c_p]])
        dR_roll_dphi = np.array([[-s_p, 0.0, c_p], [0.0, 0.0, 0.0], [-c_p, 0.0, -s_p]])
        R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, c_t, -s_t], [0.0, s_t, c_t]])
        dR_pitch_dtheta = np.array([[0.0, 0.0, 0.0], [0.0, -s_t, -c_t], [0.0, c_t, -s_t]])

        rods = [
            (self.C1_local, self.A1, self.L1, self.OFFSET_SIGN_1,
             self.ROTATION_SIGN_1, self.ALPHA1_START),
            (self.C2_local, self.A2, self.L2, self.OFFSET_SIGN_2,
             self.ROTATION_SIGN_2, self.ALPHA2_START),
        ]

        J = np.zeros((2, 2))
        for i, (C_local, A, L, offset_sign, rotation_sign, alpha_start) in enumerate(rods):
            rolled = R_roll @ C_local
            C_i = self.O + R_pitch @ rolled
            dC_dtheta = dR_pitch_dtheta @ rolled
            dC_dphi = R_pitch @ (dR_roll_dphi @ C_local)

            D = C_i - A
            alpha_i, feasible, _ = self._solve_motor_angle(
                D, self.R, L, alpha_start, offset_sign, rotation_sign
            )
            beta = rotation_sign * alpha_i  # same branch _solve_motor_angle picked
            cB, sB = np.cos(beta), np.sin(beta)
            denom = self.R * (D[2] * cB - D[1] * sB)

            if not feasible or abs(denom) < 1e-9:
                raise AnkleUnreachableError(
                    f"Rod {i + 1} at/past its kinematic singularity at "
                    f"pitch={np.degrees(theta):.2f}deg, roll={np.degrees(phi):.2f}deg "
                    f"(denom={denom:.3e})"
                )

            rod_vec = np.array([
                D[0] - offset_sign * self.t,
                D[1] - self.R * cB,
                D[2] - self.R * sB,
            ])  # C_i - B_i, magnitude L at a valid solution
            grad_D_beta = rod_vec / denom  # d(beta)/d(D), shape (3,)

            d_beta_d_theta = grad_D_beta @ dC_dtheta
            d_beta_d_phi = grad_D_beta @ dC_dphi

            J[i, 0] = rotation_sign * d_beta_d_theta  # d(alpha_i)/d(pitch)
            J[i, 1] = rotation_sign * d_beta_d_phi    # d(alpha_i)/d(roll)

        return J

    # Above this condition number, cond(J^T) is treated as "too close to a
    # singularity to trust" and pr_torque_to_motor_torque refuses to return
    # a torque rather than silently amplifying pose-measurement noise into
    # a huge/garbage motor torque command. Tune this against your actual
    # encoder noise floor and motor torque limits; 50 is a starting point,
    # not a measured value.
    MAX_JACOBIAN_CONDITION = 50.0

    def pr_torque_to_motor_torque(self, pitch_rad: float, roll_rad: float,
                                   tau_pitch: float, tau_roll: float, *,
                                   max_condition: float = MAX_JACOBIAN_CONDITION
                                   ) -> Tuple[float, float]:
        """
        Convert a desired (tau_pitch, tau_roll) [Nm] pair into the motor
        torques (tau1, tau2) [Nm] for A1/A2, via the torque-transpose of
        the analytic position Jacobian at the current pose:
        tau_alpha = (J^T)^-1 tau_pr.

        Raises AnkleUnreachableError if:
          - either rod is at/past its own kinematic singularity, or
          - cond(J^T) > max_condition, i.e. the pose is close enough to a
            singularity that small pose-measurement noise would blow up
            into a large, untrustworthy motor torque. This is a real
            physical limit of the linkage (near full rod extension), not a
            numerical artifact -- expect it to trigger at the edges of the
            reachable pitch/roll envelope.
        """
        J = self.jacobian_pr_to_motor_analytic(pitch_rad, roll_rad)

        cond = np.linalg.cond(J.T)
        if not np.isfinite(cond) or cond > max_condition:
            raise AnkleUnreachableError(
                f"PR->motor Jacobian too ill-conditioned to trust at "
                f"pitch={np.degrees(pitch_rad):.2f}deg, roll={np.degrees(roll_rad):.2f}deg "
                f"(cond={cond:.1f} > max_condition={max_condition}) -- "
                f"pose is near a kinematic singularity of the linkage."
            )

        tau_pr = np.array([tau_pitch, tau_roll], dtype=float)
        try:
            tau_motor = np.linalg.solve(J.T, tau_pr)
        except np.linalg.LinAlgError as e:
            raise AnkleUnreachableError(
                f"Singular PR->motor Jacobian at pitch={np.degrees(pitch_rad):.2f}deg, "
                f"roll={np.degrees(roll_rad):.2f}deg: {e}"
            )
        return float(tau_motor[0]), float(tau_motor[1])

    def mit_pr_to_motor_torque(
        self,
        pitch_des_rad: float, roll_des_rad: float,
        pitch_cur_rad: float, roll_cur_rad: float,
        dpitch_des: float = 0.0, droll_des: float = 0.0,
        dpitch_cur: float = 0.0, droll_cur: float = 0.0,
        kp_pitch: float = 0.0, kd_pitch: float = 0.0,
        kp_roll: float = 0.0, kd_roll: float = 0.0,
        tau_ff_pitch: float = 0.0, tau_ff_roll: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Top-level entry point: MIT-style virtual (pitch, roll) joint ->
        real A1/A2 motor feedforward torques.

        tau_pitch = kp_pitch*(pitch_des - pitch_cur) + kd_pitch*(dpitch_des - dpitch_cur) + tau_ff_pitch
        tau_roll  = kp_roll *(roll_des  - roll_cur ) + kd_roll *(droll_des  - droll_cur ) + tau_ff_roll

        then mapped to (tau1, tau2) through pr_torque_to_motor_torque,
        linearized at the CURRENT pose (pitch_cur_rad, roll_cur_rad).

        All angles in radians, angular velocities in rad/s, kp in Nm/rad,
        kd in Nm/(rad/s), tau_ff in Nm.

        Returned (tau1, tau2) are meant to be sent as the feedforward
        torque field of the A1/A2 motors' own MIT command, with kp=kd=0
        set on the motor side -- i.e. the motors run in pure torque mode,
        the PD happens up here at the pitch/roll level, not down at the
        motor level.
        """
        tau_pitch = (
            kp_pitch * (pitch_des_rad - pitch_cur_rad)
            + kd_pitch * (dpitch_des - dpitch_cur)
            + tau_ff_pitch
        )
        tau_roll = (
            kp_roll * (roll_des_rad - roll_cur_rad)
            + kd_roll * (droll_des - droll_cur)
            + tau_ff_roll
        )
        return self.pr_torque_to_motor_torque(
            pitch_cur_rad, roll_cur_rad, tau_pitch, tau_roll
        )