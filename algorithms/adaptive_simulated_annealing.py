"""ESA Stage 2 simulated annealing adapted to this project's chromosome API.

Keeps the public interface expected by variable_budget_optimization.py:
adaptive_simulated_annealing(...)->OptimizationResult

The search strategy follows the supplied Stage 2 implementation:
replace, multiswap, phase shift, residual-guided insertion, destruction/
reconstruction, reheating, and optional L-BFGS-B phase refinement.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

TAU = 2.0 * np.pi


def _phase_index(udp, orbit: int, slot: int) -> int:
    return udp.num_orbits + orbit * udp.max_per_orbit + slot


def _active_slots(solution: np.ndarray, udp) -> list[tuple[int, int]]:
    return [
        (orbit, slot)
        for orbit in range(udp.num_orbits)
        for slot in range(int(round(solution[orbit])))
    ]


def _remove_slot(solution: np.ndarray, udp, orbit: int, slot: int) -> np.ndarray:
    candidate = solution.copy()
    count = int(round(candidate[orbit]))
    start = _phase_index(udp, orbit, 0)
    phases = np.delete(candidate[start:start + count].copy(), slot)
    candidate[orbit] = count - 1
    candidate[start:start + count - 1] = phases
    candidate[start + count - 1] = 0.0
    return candidate


def _orbit_signal(udp, orbit: int, phase: float) -> np.ndarray:
    """Same single-spacecraft signal construction used by the supplied Stage 2."""
    if orbit < udp.num_dro:
        frequency = udp.frequencies_dro[orbit]
        start = (TAU - phase) / TAU / frequency
        return udp.amplitudes_dro[orbit] * udp._repeating_box_window(
            udp.t, start, frequency=frequency
        )

    if orbit < udp.num_dro + udp.num_lyap:
        local = orbit - udp.num_dro
        f1 = udp.frequencies_lyap[local]
        f2 = udp.frequencies_lyap[udp.num_lyap + local]
        return (
            udp.amplitudes_lyap[local] * udp._repeating_box_window(
                udp.t, (TAU - phase) / TAU / f1, frequency=f1
            )
            + udp.amplitudes_lyap[udp.num_lyap + local] * udp._repeating_box_window(
                udp.t, (phase + np.pi) / f2, frequency=f2
            )
        )

    local = orbit - udp.num_dro - udp.num_lyap
    f1 = udp.frequencies_axial[local]
    f2 = udp.frequencies_axial[udp.num_axial + local]
    return (
        udp.amplitudes_axial[local] * udp._repeating_box_window(
            udp.t, (TAU - phase) / TAU / f1, frequency=f1
        )
        + udp.amplitudes_axial[udp.num_axial + local] * udp._repeating_box_window(
            udp.t, (phase + np.pi) / f2, frequency=f2
        )
    )


def _mse(udp, signal_sum: np.ndarray) -> float:
    reconstructed = np.clip(signal_sum, 0.0, 1.0)
    return float(np.mean((reconstructed - udp.target_signal) ** 2))


def _to_fleet(udp, solution: np.ndarray) -> list[dict]:
    fleet = []
    for orbit, slot in _active_slots(solution, udp):
        phase = float(solution[_phase_index(udp, orbit, slot)] % TAU)
        fleet.append({
            "orbit": orbit,
            "phase": phase,
            "signal": _orbit_signal(udp, orbit, phase),
        })
    return fleet


def _to_solution(udp, fleet: list[dict]) -> np.ndarray:
    solution = np.zeros(udp.num_orbits * (udp.max_per_orbit + 1), dtype=float)
    counts = np.zeros(udp.num_orbits, dtype=int)
    for spacecraft in fleet:
        orbit = int(spacecraft["orbit"])
        slot = counts[orbit]
        if slot >= udp.max_per_orbit:
            raise ValueError(f"Orbit {orbit} exceeds maximum capacity.")
        solution[orbit] += 1
        solution[_phase_index(udp, orbit, slot)] = float(spacecraft["phase"]) % TAU
        counts[orbit] += 1
    return solution


@dataclass
class OptimizationResult:
    best_solution: np.ndarray
    objective: float
    spacecraft: int
    mse: float
    feasible: bool
    runtime: float
    iterations: int


class _Stage2Search:
    def __init__(self, udp, initial_solution: np.ndarray, seed: int):
        self.udp = udp
        self.rng = np.random.default_rng(seed)
        self.fleet = _to_fleet(udp, initial_solution)
        self.signal_sum = self._fleet_signal(self.fleet)
        self.current_mse = _mse(udp, self.signal_sum)
        self.best_fleet = copy.deepcopy(self.fleet)
        self.best_mse = self.current_mse
        self.best_feasible_fleet = (
            copy.deepcopy(self.fleet) if self.current_mse <= udp.mse_thresh else None
        )
        self.best_feasible_mse = (
            self.current_mse if self.best_feasible_fleet is not None else float("inf")
        )

    @staticmethod
    def _fleet_signal(fleet: list[dict]) -> np.ndarray:
        if not fleet:
            return np.array([], dtype=float)
        total = np.zeros_like(fleet[0]["signal"])
        for spacecraft in fleet:
            total += spacecraft["signal"]
        return total

    def _count_on_orbit(self, fleet: list[dict], orbit: int) -> int:
        return sum(int(item["orbit"] == orbit) for item in fleet)

    def _insert(self, fleet: list[dict], signal_sum: np.ndarray) -> np.ndarray:
        available = [
            orbit for orbit in range(self.udp.num_orbits)
            if self._count_on_orbit(fleet, orbit) < self.udp.max_per_orbit
        ]
        if not available:
            return signal_sum
        orbit = int(self.rng.choice(available))
        phase = self._residual_phase(orbit, signal_sum)
        signal = _orbit_signal(self.udp, orbit, phase)
        fleet.append({"orbit": orbit, "phase": phase, "signal": signal})
        return signal_sum + signal

    def _residual_phase(self, orbit: int, signal_sum: np.ndarray) -> float:
        residual = self.udp.target_signal - np.clip(signal_sum, 0.0, 1.0)
        best_phase = 0.0
        best_score = -float("inf")
        for phase in np.linspace(0.0, TAU, 50, endpoint=False):
            signal = _orbit_signal(self.udp, orbit, float(phase))
            score = float(np.dot(signal, residual))
            if score > best_score:
                best_phase, best_score = float(phase), score
        for phase in np.linspace(best_phase - 0.1, best_phase + 0.1, 10, endpoint=False):
            wrapped = float(phase % TAU)
            score = float(np.dot(_orbit_signal(self.udp, orbit, wrapped), residual))
            if score > best_score:
                best_phase, best_score = wrapped, score
        return best_phase

    def _replace(self):
        candidate = copy.deepcopy(self.fleet)
        index = int(self.rng.integers(len(candidate)))
        removed = candidate.pop(index)
        candidate_sum = self.signal_sum - removed["signal"]
        candidate_sum = self._insert(candidate, candidate_sum)
        return candidate, candidate_sum

    def _multiswap(self):
        candidate = copy.deepcopy(self.fleet)
        amount = 2 if self.rng.random() < 0.8 else 3
        amount = min(amount, len(candidate))
        indices = sorted(self.rng.choice(len(candidate), size=amount, replace=False), reverse=True)
        candidate_sum = self.signal_sum.copy()
        for index in indices:
            candidate_sum -= candidate[index]["signal"]
            del candidate[int(index)]
        for _ in range(amount):
            candidate_sum = self._insert(candidate, candidate_sum)
        return candidate, candidate_sum

    def _phase_shift(self):
        candidate = copy.deepcopy(self.fleet)
        index = int(self.rng.integers(len(candidate)))
        item = candidate[index]
        phase = (float(item["phase"]) + self.rng.normal(0.0, 0.1)) % TAU
        signal = _orbit_signal(self.udp, int(item["orbit"]), phase)
        candidate_sum = self.signal_sum - item["signal"] + signal
        item["phase"], item["signal"] = phase, signal
        return candidate, candidate_sum

    def _phase_refinement(self, fleet: list[dict]):
        if not fleet:
            empty = np.zeros_like(self.udp.target_signal)
            return fleet, empty, _mse(self.udp, empty)

        def objective(phases):
            total = np.zeros_like(self.udp.target_signal)
            for item, phase in zip(fleet, phases):
                total += _orbit_signal(self.udp, int(item["orbit"]), float(phase))
            return _mse(self.udp, total)

        initial = np.asarray([item["phase"] for item in fleet], dtype=float)
        result = minimize(
            objective, initial, method="L-BFGS-B",
            bounds=[(0.0, TAU)] * len(fleet),
            options={"maxiter": 30, "ftol": 1e-4},
        )
        refined = copy.deepcopy(fleet)
        total = np.zeros_like(self.udp.target_signal)
        for item, phase in zip(refined, result.x):
            wrapped = float(phase % TAU)
            signal = _orbit_signal(self.udp, int(item["orbit"]), wrapped)
            item["phase"], item["signal"] = wrapped, signal
            total += signal
        return refined, total, _mse(self.udp, total)

    def _maybe_refine(self, fleet, signal_sum, value):
        if value <= 1.06 * self.udp.mse_thresh:
            probability = 1.0
        elif value <= 1.20 * self.udp.mse_thresh:
            probability = 0.1
        else:
            probability = 0.0
        if self.rng.random() < probability:
            return self._phase_refinement(fleet)
        return fleet, signal_sum, value

    def _destruction_reconstruction(self, count: int):
        if len(self.fleet) <= count:
            return
        candidate = copy.deepcopy(self.fleet)
        candidate_sum = self.signal_sum.copy()
        for _ in range(count):
            index = min(
                range(len(candidate)),
                key=lambda idx: _mse(self.udp, candidate_sum - candidate[idx]["signal"]),
            )
            candidate_sum -= candidate[index]["signal"]
            del candidate[index]

        for _ in range(count):
            sampled = self.rng.choice(
                self.udp.num_orbits,
                size=min(150, self.udp.num_orbits),
                replace=False,
            )
            best = None
            best_value = float("inf")
            for orbit_value in sampled:
                orbit = int(orbit_value)
                if self._count_on_orbit(candidate, orbit) >= self.udp.max_per_orbit:
                    continue
                phase = self._residual_phase(orbit, candidate_sum)
                signal = _orbit_signal(self.udp, orbit, phase)
                value = _mse(self.udp, candidate_sum + signal)
                if value < best_value:
                    best = (orbit, phase, signal)
                    best_value = value
            if best is None:
                candidate_sum = self._insert(candidate, candidate_sum)
            else:
                orbit, phase, signal = best
                candidate.append({"orbit": orbit, "phase": phase, "signal": signal})
                candidate_sum += signal
        self.fleet, self.signal_sum = candidate, candidate_sum
        self.current_mse = _mse(self.udp, candidate_sum)

    def _record_best(self):
        improved = False
        if self.current_mse < self.best_mse:
            self.best_mse = self.current_mse
            self.best_fleet = copy.deepcopy(self.fleet)
            improved = True
        if self.current_mse <= self.udp.mse_thresh and self.current_mse < self.best_feasible_mse:
            self.best_feasible_mse = self.current_mse
            self.best_feasible_fleet = copy.deepcopy(self.fleet)
            improved = True
        return improved

    def run(self, iterations: int, log_every: int):
        temperature = 0.5
        temperature_min = 1e-6
        cooling_rate = 0.999995
        steps_since_improvement = 0
        destruction_period = 300_000
        reheat_after = 100_000

        for iteration in range(1, iterations + 1):
            if iteration % destruction_period == 0:
                self._destruction_reconstruction(int(self.rng.choice((3, 4, 5))))
                self._record_best()

            draw = self.rng.random()
            if draw < 0.4:
                candidate, candidate_sum = self._replace()
            elif draw < 0.5:
                candidate, candidate_sum = self._multiswap()
            else:
                candidate, candidate_sum = self._phase_shift()

            candidate_mse = _mse(self.udp, candidate_sum)
            candidate, candidate_sum, candidate_mse = self._maybe_refine(
                candidate, candidate_sum, candidate_mse
            )
            delta = candidate_mse - self.current_mse
            accepted = delta < 0.0 or self.rng.random() < math.exp(
                -delta / max(temperature, temperature_min)
            )
            if accepted:
                self.fleet, self.signal_sum, self.current_mse = (
                    candidate, candidate_sum, candidate_mse
                )
                if self._record_best():
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

            temperature = max(temperature_min, temperature * cooling_rate)
            if steps_since_improvement >= reheat_after and temperature < 0.05:
                temperature = 0.5
                steps_since_improvement = 0

            if log_every and iteration % log_every == 0:
                print(
                    f"Iteration {iteration}/{iterations} | T={temperature:.6g} "
                    f"| N={len(self.fleet)} | Current MSE={self.current_mse:.8f} "
                    f"| Best MSE={self.best_mse:.8f}"
                )

        return self.best_feasible_fleet or self.best_fleet


def adaptive_simulated_annealing(
    udp,
    initial_solution: np.ndarray,
    iterations: int = 10_000,
    seed: int = 42,
    refine_every: int = 2_000,
    log_every: int = 1_000,
) -> OptimizationResult:
    """Run the supplied ESA Stage 2 strategy on this project's chromosome."""
    started = time.time()
    initial = np.asarray(initial_solution, dtype=float).copy()
    expected = udp.num_orbits * (udp.max_per_orbit + 1)
    if initial.size != expected:
        raise ValueError(f"Expected solution length {expected}, got {initial.size}.")
    initial[:udp.num_orbits] = np.rint(initial[:udp.num_orbits])

    _, _, initial_official_mse = udp.fitness(initial, postprocess=True)
    if float(initial_official_mse) > udp.mse_thresh:
        raise ValueError(
            f"Initial solution must be feasible; official MSE={float(initial_official_mse):.12f}"
        )

    search = _Stage2Search(udp, initial, seed)
    budget = len(search.fleet)
    print("\nESA STAGE 2 SA (ADAPTED)")
    print("------------------------")
    print(f"Initial fleet: {budget} spacecraft | MSE: {search.current_mse:.12f}")

    fleet = search.run(iterations, log_every)
    best_solution = _to_solution(udp, fleet)
    objective, spacecraft, official_mse = udp.fitness(best_solution, postprocess=True)

    # Keep the returned candidate only if the official evaluator agrees it is feasible.
    if float(official_mse) > udp.mse_thresh:
        best_solution = initial
        objective, spacecraft, official_mse = udp.fitness(best_solution, postprocess=True)

    return OptimizationResult(
        best_solution=best_solution,
        objective=float(objective[0]),
        spacecraft=int(spacecraft),
        mse=float(official_mse),
        feasible=float(official_mse) <= udp.mse_thresh,
        runtime=time.time() - started,
        iterations=iterations,
    )
