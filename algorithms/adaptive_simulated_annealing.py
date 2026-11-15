"""Fixed-budget fleet refinement using residual-guided simulated annealing.

The fleet size is kept constant. Moves replace spacecraft/orbits or perturb phases;
phase candidates are guided by the residual between target and current signal.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


TAU = 2.0 * np.pi


def _raw_fleet_signal(udp, solution: np.ndarray) -> np.ndarray:
    """Sum individual spacecraft signals without allocating ESA's large matrix."""
    raw = np.zeros_like(udp.target_signal, dtype=float)
    for orbit in range(udp.num_orbits):
        count = int(round(solution[orbit]))
        for slot in range(count):
            phase = solution[_phase_index(udp, orbit, slot)]
            raw += _orbit_signal(udp, orbit, float(phase))
    return raw


def _mse_from_raw(udp, raw_signal: np.ndarray) -> float:
    reconstructed = np.clip(raw_signal, 0.0, 1.0)
    return float(np.mean((reconstructed - udp.target_signal) ** 2))


def _orbit_signal(udp, orbit: int, phase: float) -> np.ndarray:
    """Return the exact contribution of one spacecraft on the given orbit."""
    t = udp.t
    if orbit < udp.num_dro:
        freq = udp.frequencies_dro[orbit]
        center = (TAU - phase) / TAU / freq
        return udp.amplitudes_dro[orbit] * udp._repeating_box_window(
            t, center, freq
        )

    if orbit < udp.num_dro + udp.num_lyap:
        local = orbit - udp.num_dro
        f1 = udp.frequencies_lyap[local]
        f2 = udp.frequencies_lyap[udp.num_lyap + local]
        c1 = (TAU - phase) / TAU / f1
        c2 = (phase + np.pi) / f2
        return (
            udp.amplitudes_lyap[local]
            * udp._repeating_box_window(t, c1, f1)
            + udp.amplitudes_lyap[udp.num_lyap + local]
            * udp._repeating_box_window(t, c2, f2)
        )

    local = orbit - udp.num_dro - udp.num_lyap
    f1 = udp.frequencies_axial[local]
    f2 = udp.frequencies_axial[udp.num_axial + local]
    c1 = (TAU - phase) / TAU / f1
    c2 = (phase + np.pi) / f2
    return (
        udp.amplitudes_axial[local]
        * udp._repeating_box_window(t, c1, f1)
        + udp.amplitudes_axial[udp.num_axial + local]
        * udp._repeating_box_window(t, c2, f2)
    )


def _active_slots(solution: np.ndarray, udp) -> list[tuple[int, int]]:
    slots = []
    for orbit in range(udp.num_orbits):
        for slot in range(int(round(solution[orbit]))):
            slots.append((orbit, slot))
    return slots


def _phase_index(udp, orbit: int, slot: int) -> int:
    return udp.num_orbits + orbit * udp.max_per_orbit + slot


def _remove_slot(
    solution: np.ndarray, udp, orbit: int, slot: int
) -> np.ndarray:
    candidate = solution.copy()
    count = int(round(candidate[orbit]))
    start = _phase_index(udp, orbit, 0)
    phases = candidate[start:start + count].copy()
    phases = np.delete(phases, slot)
    candidate[orbit] = count - 1
    candidate[start:start + count - 1] = phases
    candidate[start + count - 1] = 0.0
    return candidate


def _add_to_orbit(
    solution: np.ndarray, udp, orbit: int, phase: float
) -> np.ndarray:
    candidate = solution.copy()
    count = int(round(candidate[orbit]))
    if count >= udp.max_per_orbit:
        return candidate
    candidate[orbit] = count + 1
    candidate[_phase_index(udp, orbit, count)] = phase % TAU
    return candidate


def _residual_phase(
    udp,
    orbit: int,
    current_signal: np.ndarray,
    rng: np.random.Generator,
    grid_size: int = 50,
) -> float:
    residual = udp.target_signal - current_signal
    grid = np.linspace(0.0, TAU, grid_size, endpoint=False)
    scores = [
        float(np.dot(_orbit_signal(udp, orbit, p), residual))
        for p in grid
    ]
    best = float(grid[int(np.argmax(scores))])

    local = best + np.linspace(-0.10, 0.10, 11)
    local_scores = [
        float(
            np.dot(
                _orbit_signal(udp, orbit, p % TAU),
                residual,
            )
        )
        for p in local
    ]
    best = float(local[int(np.argmax(local_scores))] % TAU)

    # Small randomized jitter avoids repeatedly proposing the same phase.
    return float((best + rng.normal(0.0, 0.015)) % TAU)


def _official_mse(udp, solution: np.ndarray) -> float:
    _, _, mse = udp.fitness(solution, postprocess=True)
    return float(mse)


def _phase_refine(
    udp, solution: np.ndarray, maxiter: int = 25
) -> np.ndarray:
    """Locally refine active phases while keeping orbit counts unchanged."""
    slots = _active_slots(solution, udp)
    if not slots:
        return solution.copy()

    indices = np.array(
        [_phase_index(udp, o, s) for o, s in slots],
        dtype=int,
    )
    x0 = solution[indices].copy()

    base_raw = _raw_fleet_signal(udp, solution)
    old_signals = [
        _orbit_signal(
            udp,
            orbit,
            float(solution[_phase_index(udp, orbit, slot)]),
        )
        for orbit, slot in slots
    ]

    def objective(active_phases):
        raw = base_raw.copy()
        for (orbit, slot), old_signal, phase in zip(
            slots, old_signals, active_phases
        ):
            raw += (
                _orbit_signal(udp, orbit, float(phase % TAU))
                - old_signal
            )
        return _mse_from_raw(udp, raw)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(0.0, TAU) for _ in indices],
        options={"maxiter": maxiter, "ftol": 1e-7},
    )

    refined = solution.copy()
    refined[indices] = np.mod(result.x, TAU)
    return refined


@dataclass
class OptimizationResult:
    best_solution: np.ndarray
    objective: float
    spacecraft: int
    mse: float
    feasible: bool
    runtime: float
    iterations: int


def adaptive_simulated_annealing(
    udp,
    initial_solution: np.ndarray,
    iterations: int = 10_000,
    seed: int = 42,
    refine_every: int = 2_000,
    log_every: int = 1_000,
) -> OptimizationResult:
    """Improve a starting fleet without changing its spacecraft budget."""
    start_time = time.time()
    rng = np.random.default_rng(seed)

    current = np.asarray(initial_solution, dtype=float).copy()
    expected = udp.num_orbits * (udp.max_per_orbit + 1)

    if current.size != expected:
        raise ValueError(
            f"Expected solution length {expected}, got {current.size}."
        )

    # Canonicalize count fields and validate the supplied fleet with the evaluator.
    current[:udp.num_orbits] = np.rint(
        current[:udp.num_orbits]
    )

    current_raw = _raw_fleet_signal(udp, current)
    current_mse = _mse_from_raw(udp, current_raw)

    best = current.copy()
    best_mse = current_mse
    budget = int(round(current[:udp.num_orbits].sum()))

    temperature = 0.05
    min_temperature = 1e-5
    cooling = 0.9995
    stale = 0

    print("\nADAPTIVE SIMULATED ANNEALING")
    print("----------------------------")
    print(
        f"Initial fleet: {budget} spacecraft | "
        f"MSE: {current_mse:.12f}"
    )

    for step in range(1, iterations + 1):
        candidate = current.copy()
        candidate_raw = current_raw.copy()
        slots = _active_slots(current, udp)

        if not slots:
            break

        move = rng.random()

        if move < 0.50:
            # Replace one spacecraft with one on a different available orbit.
            orbit, slot = slots[int(rng.integers(len(slots)))]
            old_phase = float(
                current[_phase_index(udp, orbit, slot)]
            )

            candidate_raw -= _orbit_signal(
                udp, orbit, old_phase
            )

            reduced = _remove_slot(
                current, udp, orbit, slot
            )

            available = np.flatnonzero(
                reduced[:udp.num_orbits] < udp.max_per_orbit
            )

            if available.size:
                new_orbit = int(rng.choice(available))
                phase = _residual_phase(
                    udp,
                    new_orbit,
                    np.clip(candidate_raw, 0.0, 1.0),
                    rng,
                )
                candidate = _add_to_orbit(
                    reduced, udp, new_orbit, phase
                )
                candidate_raw += _orbit_signal(
                    udp, new_orbit, phase
                )

        elif move < 0.72 and len(slots) >= 2:
            # Multi-swap: remove two or three and reconstruct the same number.
            amount = min(
                len(slots),
                3 if rng.random() < 0.2 else 2,
            )

            chosen = rng.choice(
                len(slots),
                size=amount,
                replace=False,
            )

            removals = [
                slots[int(i)] for i in chosen
            ]

            candidate = current.copy()

            # Remove descending slot order within each orbit to keep indices valid.
            for orbit in sorted({o for o, _ in removals}):
                orbit_removals = sorted(
                    (
                        x for x in removals
                        if x[0] == orbit
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for _, slot in orbit_removals:
                    old_phase = float(
                        candidate[_phase_index(udp, orbit, slot)]
                    )
                    candidate_raw -= _orbit_signal(
                        udp, orbit, old_phase
                    )
                    candidate = _remove_slot(
                        candidate, udp, orbit, slot
                    )

            for _ in range(amount):
                available = np.flatnonzero(
                    candidate[:udp.num_orbits] < udp.max_per_orbit
                )

                if not available.size:
                    break

                new_orbit = int(rng.choice(available))
                phase = _residual_phase(
                    udp,
                    new_orbit,
                    np.clip(candidate_raw, 0.0, 1.0),
                    rng,
                )

                candidate = _add_to_orbit(
                    candidate, udp, new_orbit, phase
                )

                candidate_raw += _orbit_signal(
                    udp, new_orbit, phase
                )

        else:
            # Local phase mutation.
            orbit, slot = slots[int(rng.integers(len(slots)))]
            idx = _phase_index(udp, orbit, slot)

            old_phase = float(candidate[idx])
            sigma = max(
                0.015,
                0.20 * (temperature / 0.05),
            )

            candidate[idx] = (
                candidate[idx] + rng.normal(0.0, sigma)
            ) % TAU

            candidate_raw += (
                _orbit_signal(
                    udp, orbit, float(candidate[idx])
                )
                - _orbit_signal(udp, orbit, old_phase)
            )

        # Guard the fixed-budget invariant.
        if int(round(candidate[:udp.num_orbits].sum())) != budget:
            continue

        candidate_mse = _mse_from_raw(
            udp, candidate_raw
        )

        delta = candidate_mse - current_mse

        if (
            delta <= 0.0
            or rng.random()
            < math.exp(
                -delta / max(temperature, min_temperature)
            )
        ):
            current = candidate
            current_mse = candidate_mse
            current_raw = candidate_raw

        if current_mse < best_mse:
            best = current.copy()
            best_mse = current_mse
            stale = 0
        else:
            stale += 1

        temperature = max(
            min_temperature,
            temperature * cooling,
        )

        if stale >= 2_500:
            temperature = max(temperature, 0.02)
            stale = 0

        if refine_every and step % refine_every == 0:
            refined = _phase_refine(
                udp, best, maxiter=20
            )

            refined_mse = _mse_from_raw(
                udp,
                _raw_fleet_signal(udp, refined),
            )

            if refined_mse < best_mse:
                best = refined
                best_mse = refined_mse
                current = refined.copy()
                current_mse = refined_mse
                current_raw = _raw_fleet_signal(
                    udp, current
                )

        if log_every and step % log_every == 0:
            print(
                f"Iteration {step}/{iterations} | "
                f"Best MSE: {best_mse:.12f} | "
                f"T: {temperature:.6g}"
            )

    # Final local phase refinement; keep it only if it improves official MSE.
    refined = _phase_refine(
        udp, best, maxiter=40
    )

    refined_mse = _mse_from_raw(
        udp,
        _raw_fleet_signal(udp, refined),
    )

    if refined_mse < best_mse:
        best = refined
        best_mse = refined_mse

    objective, spacecraft, official_mse = udp.fitness(
        best, postprocess=True
    )

    return OptimizationResult(
        best_solution=best,
        objective=float(objective[0]),
        spacecraft=int(spacecraft),
        mse=float(official_mse),
        feasible=float(official_mse) <= udp.mse_thresh,
        runtime=time.time() - start_time,
        iterations=iterations,
    )