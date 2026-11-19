"""CUDA-Q statevector and molecular demonstrations for RCDS + MetaMap-ML.

H2 energies use PySCF STO-3G integrals, Jordan-Wigner mapping, and a four-qubit
UCCSD ansatz.  Results are research simulations, not clinical predictions.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="cudaq")
DEFAULT_MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIG))

import cudaq
import cudaq.kernels
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

try:
    from pyscf import gto, scf
except ImportError as exc:  # pragma: no cover - installation guidance
    raise ImportError("H2 STO-3G requires PySCF; install with `pip install pyscf`.") from exc


def _target_names() -> set[str]:
    return {str(target.name) for target in cudaq.get_targets()}


def select_target() -> str:
    """Select NVIDIA/cuStateVec when compiled and available; otherwise qpp-cpu."""
    names = _target_names()
    gpu_available = bool(cudaq.has_target("nvidia")) and "nvidia" in names
    selected = "nvidia" if gpu_available else "qpp-cpu"
    if selected not in names:
        raise RuntimeError(f"CUDA-Q target {selected!r} is unavailable")
    cudaq.set_target(selected)
    return selected


def backend_status() -> dict[str, str | bool]:
    names = _target_names()
    available = bool(cudaq.has_target("nvidia")) and "nvidia" in names
    active = select_target()
    return {"target": active,
            "backend": "NVIDIA GPU · cuStateVec" if available else "CPU · qpp-cpu",
            "nvidia_available": available,
            "status": "GPU target available" if available else "CPU fallback · NVIDIA target unavailable"}


@cudaq.kernel
def ghz_chain(qubit_count: int):
    """Prepare |GHZ_n> using H on q0 and nearest-neighbor CNOTs."""
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for index in range(qubit_count - 1):
        x.ctrl(qubits[index], qubits[index + 1])


def ghz_statevector_probabilities(qubit_count: int) -> dict[str, float]:
    """Return the two GHZ statevector probabilities, checking the ideal state."""
    select_target()
    state = cudaq.get_state(ghz_chain, int(qubit_count))
    vector = np.asarray(state.to_numpy(), dtype=complex).reshape(-1)
    probs = np.abs(vector) ** 2
    zero, one = float(probs[0]), float(probs[-1])
    if not np.allclose([zero, one], [0.5, 0.5], atol=1e-7) or probs.sum() - zero - one > 1e-7:
        raise RuntimeError("GHZ statevector failed the 50/50 endpoint-state check")
    return {"0" * qubit_count: zero, "1" * qubit_count: one}


def _measure_ghz_target(target: str, sizes: range) -> list[tuple[int, float]]:
    cudaq.set_target(target)
    results = []
    for n in sizes:
        # Materialize the full statevector (the requested scaling workload).
        cudaq.get_state(ghz_chain, n)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            state = cudaq.get_state(ghz_chain, n)
            state.to_numpy()
            samples.append(time.perf_counter() - start)
        results.append((n, float(np.median(samples))))
    return results


def benchmark_ghz_comparison() -> dict:
    """Compare measured CPU/GPU statevector timings, or label GPU projection."""
    status = backend_status()
    sizes = range(2, 21)
    names = _target_names()
    if "qpp-cpu" not in names:
        raise RuntimeError("qpp-cpu is required for the portable scaling baseline")
    cpu = _measure_ghz_target("qpp-cpu", sizes)
    if status["nvidia_available"]:
        gpu = _measure_ghz_target("nvidia", sizes)
        gpu_label = "Measured NVIDIA cuStateVec"
        projected = False
    else:
        # Extrapolation for display only; do not present it as measured GPU data.
        ns = np.asarray([n for n, _ in cpu], dtype=float)
        cpu_seconds = np.asarray([t for _, t in cpu])
        speedup = 1.5 + (ns - 2.0) * (13.5 / 18.0)
        gpu = [(int(n), float(t / factor)) for n, t, factor in zip(ns, cpu_seconds, speedup)]
        gpu_label = "Projected cuStateVec (GPU unavailable)"
        projected = True
    select_target()

    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    ax.plot(*zip(*cpu), marker="o", color="#42bfd1", label="Measured qpp-cpu")
    ax.plot(*zip(*gpu), marker="s", linestyle="--" if projected else "-",
            color="#50d6a3", label=gpu_label)
    ax.set_yscale("log")
    ax.set_xlabel("Qubit count")
    ax.set_ylabel("Statevector runtime (s, log scale)")
    ax.set_title("GHZ statevector scaling · 2–20 qubits")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "ghz_scaling.png", dpi=160)
    plt.close(fig)

    print(f"CUDA-Q target: {status['target']} · {status['backend']}")
    print("\nGHZ statevector runtime · median of 3 runs; first execution excluded")
    print(f"{'Qubits':>7} {'qpp-cpu measured (s)':>22} {gpu_label:>40}")
    for (n, cpu_s), (_, gpu_s) in zip(cpu, gpu):
        print(f"{n:>7} {cpu_s:>22.6g} {gpu_s:>40.6g}")
    return {"cpu": cpu, "gpu": gpu, "gpu_label": gpu_label,
            "gpu_projected": projected, "target": status["target"],
            "backend": status["backend"]}


def benchmark_ghz() -> list[tuple[int, float]]:
    """Compatibility wrapper returning the measured active-backend timings."""
    data = benchmark_ghz_comparison()
    return data["gpu"] if not data["gpu_projected"] else data["cpu"]


def _fermion_sign(state: int, mode: int) -> int:
    return -1 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1


def _apply_fermion(state: int, mode: int, create: bool):
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    return state ^ (1 << mode), _fermion_sign(state, mode)


def _electronic_hamiltonian(mol):
    """Construct the 16x16 four-spin-orbital matrix from RHF STO-3G integrals."""
    mf = scf.RHF(mol).run(verbose=0)
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = mol.intor("int2e", aosym="s1")
    eri = np.einsum("pi,qj,rk,sl,pqrs->ijkl", mf.mo_coeff, mf.mo_coeff,
                    mf.mo_coeff, mf.mo_coeff, eri, optimize=True)
    matrix = np.zeros((16, 16), dtype=complex)

    def apply_ops(column: int, operations):
        state, amplitude = column, 1.0
        # Operators are specified in written order, applied right-to-left.
        for mode, create in reversed(operations):
            result = _apply_fermion(state, mode, create)
            if result is None:
                return None
            state, sign = result
            amplitude *= sign
        return state, amplitude

    for p in range(2):
        for q in range(2):
            for spin in range(2):
                coeff = h1[p, q]
                if abs(coeff) < 1e-14: continue
                for col in range(16):
                    result = apply_ops(col, [(2*p+spin, True), (2*q+spin, False)])
                    if result: matrix[result[0], col] += coeff * result[1]
    for p, q, r, s in np.ndindex(2, 2, 2, 2):
        coeff = 0.5 * eri[p, q, r, s]
        if abs(coeff) < 1e-14: continue
        for spin in range(2):
            for other in range(2):
                ops = [(2*p+spin, True), (2*r+other, True),
                       (2*s+other, False), (2*q+spin, False)]
                for col in range(16):
                    result = apply_ops(col, ops)
                    if result: matrix[result[0], col] += coeff * result[1]
    matrix += np.eye(16) * mol.energy_nuc()
    return (matrix + matrix.conj().T) / 2


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def _pauli_terms(matrix: np.ndarray, threshold: float = 1e-9):
    """Pauli-decompose the JW matrix; return (coefficient, q0..q3 word)."""
    terms = []
    for labels in np.ndindex(4, 4, 4, 4):
        word = "".join("IXYZ"[label] for label in labels)
        # Qubit zero is the least-significant computational-basis bit.
        operator = np.array([[1]], dtype=complex)
        for q in reversed(range(4)):
            operator = np.kron(operator, _PAULI[word[q]])
        coefficient = np.trace(operator.conj().T @ matrix) / 16.0
        if abs(coefficient) > threshold:
            terms.append((float(coefficient.real), word))
    return terms


def _cudaq_spin_operator(terms):
    result = 0.0 * cudaq.spin.i(0)
    factors = {"I": cudaq.spin.i, "X": cudaq.spin.x,
               "Y": cudaq.spin.y, "Z": cudaq.spin.z}
    for coefficient, word in terms:
        term = coefficient * cudaq.spin.i(0)
        for q, label in enumerate(word):
            if label != "I": term *= factors[label](q)
        result += term
    return result


@cudaq.kernel
def h2_uccsd(parameters: list[float]):
    """Four-qubit UCCSD, initialized in the two-electron HF determinant |1100>."""
    qubits = cudaq.qvector(4)
    x(qubits[0])
    x(qubits[1])
    cudaq.kernels.uccsd(qubits, parameters, 2, 4)


def h2_hamiltonian(distance: float):
    """Return the 15-term STO-3G H2 Jordan-Wigner Hamiltonian and dense matrix."""
    r = float(distance)
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", unit="Angstrom",
                charge=0, spin=0, verbose=0)
    matrix = _electronic_hamiltonian(mol)
    terms = _pauli_terms(matrix)
    return _cudaq_spin_operator(terms), matrix, terms


def run_vqe_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the STO-3G H2 UCCSD-VQE and exact curves over 0.3–2.5 Å."""
    select_target()
    distances = np.unique(np.r_[np.round(np.arange(0.3, 2.501, 0.1), 2), 0.74])
    vqe_energies, exact_energies, term_counts = [], [], []
    parameters = np.zeros(cudaq.kernels.uccsd_num_parameters(2, 4), dtype=float)
    for distance in distances:
        hamiltonian, matrix, terms = h2_hamiltonian(float(distance))
        sector = [state for state in range(16) if state.bit_count() == 2]
        exact = float(np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])[0])

        def objective(theta):
            return float(cudaq.observe(h2_uccsd, hamiltonian, theta.tolist()).expectation())

        result = minimize(objective, parameters, method="COBYLA",
                          options={"maxiter": 140, "rhobeg": 0.22, "tol": 1e-7})
        parameters = np.asarray(result.x, dtype=float)
        energy = min(float(result.fun), objective(parameters))
        vqe_energies.append(energy)
        exact_energies.append(exact)
        term_counts.append(len(terms))

    vqe, exact = np.asarray(vqe_energies), np.asarray(exact_energies)
    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    ax.plot(distances, exact, "--", color="#91a3b2", label="Exact diagonalization")
    ax.plot(distances, vqe, "o-", ms=3.5, color="#39c5d6", label="CUDA-Q UCCSD VQE")
    ax.axvline(0.74, color="#54d6a5", linestyle=":", label="Equilibrium ≈ 0.74 Å")
    ax.set_xlabel("H–H bond distance (Å)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("H₂ STO-3G · Jordan–Wigner · UCCSD")
    ax.grid(True, alpha=0.23)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "vqe_pes_curve.png", dpi=160)
    plt.close(fig)

    print("\nH2 STO-3G / Jordan-Wigner / CUDA-Q UCCSD VQE")
    print(f"{'R (Å)':>8} {'VQE (Ha)':>14} {'Exact (Ha)':>14} {'|error|':>12} {'Pauli terms':>13}")
    for r, v, e, count in zip(distances, vqe, exact, term_counts):
        print(f"{r:>8.2f} {v:>14.8f} {e:>14.8f} {abs(v-e):>12.3g} {count:>13}")
    if not np.all(np.asarray(term_counts) == 15):
        warnings.warn("Expected 15 Pauli terms; check Hamiltonian truncation or integral convention")
    return distances, vqe, exact


@cudaq.kernel
def metabolic_classifier(carcinogen_angle: float, galt_angle: float,
                         atp_angle: float, udpga_angle: float):
    """Encode four normalized RCDS/MetaMap features into a four-qubit circuit."""
    qubits = cudaq.qvector(4)
    ry(carcinogen_angle, qubits[0])
    ry(galt_angle, qubits[1])
    ry(atp_angle, qubits[2])
    ry(udpga_angle, qubits[3])
    x.ctrl(qubits[0], qubits[1])
    x.ctrl(qubits[1], qubits[2])
    x.ctrl(qubits[2], qubits[3])
    x.ctrl(qubits[3], qubits[0])
    rz(carcinogen_angle * galt_angle, qubits[0])
    rz(atp_angle * udpga_angle, qubits[2])


def quantum_adduct_susceptibility(carcinogen_uM: float, galt_activity: float,
                                  atp_ratio: float, udp_glucuronate: float) -> float:
    values = np.clip([carcinogen_uM / 100.0, galt_activity, atp_ratio,
                      udp_glucuronate], 0.0, 1.0)
    angles = (np.asarray(values) * np.pi).tolist()
    z_expectation = float(cudaq.observe(metabolic_classifier, cudaq.spin.z(0), *angles).expectation())
    return float(np.clip((1.0 - z_expectation) / 2.0, 0.0, 1.0))


def main():
    status = backend_status()
    print(f"CUDA-Q version: {getattr(cudaq, '__version__', 'unknown')}")
    print(f"Active target: {status['target']} · {status['backend']}")
    print(f"Target status: {status['status']}")
    probs = ghz_statevector_probabilities(8)
    print(f"8-qubit GHZ statevector check: {probs}")
    benchmark_ghz_comparison()
    run_vqe_curve()
    score = quantum_adduct_susceptibility(60.0, 0.02, 0.72, 0.08)
    print("\nQuantum-metabolic classifier demonstration")
    print("Input: carcinogen=60 uM, GALT=2%, ATP ratio=0.72, UDP-glucuronate=0.08")
    print(f"Illustrative adduct susceptibility score: {score:.4f}")
    print("Saved ghz_scaling.png and vqe_pes_curve.png beside this script")


if __name__ == "__main__":
    main()
