"""ЛР № 6. Кодирование, конечные автоматы и марковская цепь."""
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from viewer import show_gallery

OUT = Path(__file__).resolve().parent / "results"
ALPHABET = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
MOORE_Y = np.array([0, 0, 0, 1, 1, 1, 0])
RANDOM_Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
MOORE_STARTS = (0, 1, 3, 6)
STEPS = 1000
MAX_LAG = 60


def encode(text):
    """Отображает каждую букву в отдельный пятибитный блок."""
    text = text.upper()
    if any(letter not in ALPHABET for letter in text):
        raise ValueError("Допустимы только 32 буквы русского алфавита без Ё.")
    return " ".join(f"{ALPHABET.index(letter):05b}" for letter in text)


def decode(codes):
    """Восстанавливает буквы из пятибитных блоков, разделённых пробелами."""
    words = codes.split()
    if any(len(word) != 5 or set(word) - {"0", "1"} for word in words):
        raise ValueError("Каждый код должен состоять из пяти двоичных цифр.")
    return "".join(ALPHABET[int(word, 2)] for word in words)


def moore_matrix():
    """Строки — прежние состояния, столбцы — входы 0, 1, 2."""
    states = np.arange(7)
    return np.column_stack([np.zeros_like(states), (states + 1) % 7, (states - 1) % 7])


def moore(signal, initial, transitions):
    """На каждом такте сначала выполняет переход, затем вычисляет выход."""
    trajectory = np.empty(len(signal), dtype=int)
    state = initial
    for i, symbol in enumerate(signal):
        state = transitions[state, symbol]
        trajectory[i] = state
    return trajectory, MOORE_Y[trajectory]


def random_matrices():
    """P[x, z, j] — вероятность из z перейти в j при входе x."""
    # С вероятностью 0.2 делаем равномерный сброс, поэтому любая пара
    # состояний связана положительной вероятностью уже за один шаг.
    matrix = np.full((2, 10, 10), .02)
    states = np.arange(10)
    matrix[0, states, states] += .8
    matrix[1, states, (states + 1) % 10] += .8
    return matrix


def random_trajectory(signal, initial, matrices, noise):
    """Выбор состояния по накопленным вероятностям; noise содержит U[0, 1)."""
    thresholds = np.cumsum(matrices, axis=-1)
    thresholds[..., -1] = 1.0  # Защита последнего порога от округления.
    trajectory = np.empty(len(signal), dtype=int)
    state = initial
    for i, (symbol, u) in enumerate(zip(signal, noise)):
        # Считаем, сколько правых границ интервалов уже пройдено.
        state = np.searchsorted(thresholds[symbol, state], u, side="right")
        trajectory[i] = state
    return trajectory


def markov_matrix():
    """Цепь без входа: переход вперёд по циклу или выбор по весам 1…10."""
    weights = np.arange(1, 11) / 55
    matrix = np.tile(.25 * weights, (10, 1))
    states = np.arange(10)
    matrix[states, (states + 1) % 10] += .75
    return matrix


def stationary(matrix):
    """Решает Pᵀπ = π с условием суммы вероятностей, равной единице."""
    equations = matrix.T - np.eye(len(matrix))
    # Строку заменяем ПОСЛЕ вычитания I, сохраняя точное условие нормировки.
    equations[-1, :] = 1
    right = np.zeros(len(matrix))
    right[-1] = 1
    return np.linalg.solve(equations, right)


def signal_stats(signal, max_lag=MAX_LAG):
    """Смещённая АКФ без центрирования и односторонний энергетический спектр."""
    n = len(signal)
    values = np.asarray(signal, dtype=float)
    # R[k] = sum(y[t] * y[t+k]) / N. При k=0 это средняя мощность.
    corr = np.correlate(values, values, mode="full")[n-1:n+max_lag] / n
    frequencies = np.fft.rfftfreq(n)
    # E[k] = |FFT[k]|²/N. Удваиваем парные частоты, сумма E[k] = Σy².
    energy = np.abs(np.fft.rfft(values))**2 / n
    energy[1:(-1 if n % 2 == 0 else None)] *= 2
    return corr, frequencies, energy, np.bincount(signal, minlength=2) / n


def figure():
    return Figure(figsize=(10.3, 6), layout="constrained")


def coding_plot():
    fig = figure()
    ax = fig.subplots()
    ax.axis("off")
    demo = "МОДЕЛЬ"
    ax.set_title(f"{demo}  →  {encode(demo)}  →  {decode(encode(demo))}", pad=15)
    rows = [[f"{ALPHABET[i]:s}   {i:05b}" for i in range(row, 32, 8)] for row in range(8)]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(.86, 2.0)
    for (row, _), cell in table.get_celld().items():
        cell.set_facecolor("#eff6ff" if row % 2 == 0 else "#ffffff")
        cell.set_edgecolor("#cbd5e1")
    return fig


def moore_graph():
    fig = figure()
    ax = fig.subplots()
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.25, 1.25), aspect="equal",
           title="Граф переходов Мура и выходы состояний")
    ax.axis("off")
    angle = np.pi/2 - np.arange(7) * 2*np.pi/7
    points = np.c_[np.cos(angle), np.sin(angle)]
    for state, point in enumerate(points):
        ax.text(*point, f"z{state+1} / {MOORE_Y[state]}", ha="center", va="center",
                fontsize=10, zorder=5, bbox=dict(boxstyle="round,pad=.4", fc="#e0f2fe", ec="#0369a1"))
        for shift, color, rad in [(1, "#2563eb", .25), (-1, "#d97706", .25)]:
            ax.annotate("", points[(state+shift) % 7], point, zorder=2,
                        arrowprops=dict(arrowstyle="->", color=color, shrinkA=27, shrinkB=27,
                                        connectionstyle=f"arc3,rad={rad}"))
        if state != 0:
            ax.annotate("", points[0], point, zorder=1,
                        arrowprops=dict(arrowstyle="->", color="#64748b", linestyle=":",
                                        shrinkA=27, shrinkB=27, connectionstyle="arc3,rad=.1"))
    ax.annotate("", (-.13, 1.02), (.13, 1.02),
                arrowprops=dict(arrowstyle="->", color="#64748b", linestyle=":",
                                connectionstyle="arc3,rad=-1.4"))
    for i, (color, label) in enumerate([("#64748b", "0 — сброс"), ("#2563eb", "1 — вперёд"),
                                        ("#d97706", "2 — назад")]):
        ax.text(1.10, .32 - .15*i, label, color=color, fontsize=9)
    return fig


def snapshot(signal, states, output, title):
    fig = figure()
    axes = fig.subplots(3, 1, sharex=True)
    time = np.arange(1, 81)
    for ax, values, label, color in zip(axes, (signal, states+1, output),
                                        ("Вход x", "Состояние z", "Выход y"),
                                        ("#2563eb", "#0f766e", "#d97706")):
        ax.step(time, values[:80], where="post", color=color)
        ax.set_ylabel(label)
        ax.set_yticks(np.arange(values.min(), values.max()+1))
        ax.grid(alpha=.2)
    axes[-1].set(xlabel="Такт", xlim=(1, 80))
    fig.suptitle(title)
    return fig


def comparison(runs, labels, title):
    fig = figure()
    corr_ax, energy_ax, counts_ax, dist_ax = fig.subplots(2, 2).flat
    stats = [signal_stats(run) for run in runs]
    for i, (label, (corr, freq, energy, counts)) in enumerate(zip(labels, stats)):
        color = f"C{i}"
        corr_ax.plot(np.arange(len(corr)), corr, label=label, color=color,
                     linewidth=1.1, alpha=.85)
        # Постоянную компоненту f=0 сохраняем отдельно в числах и на графике.
        energy_ax.plot(freq[1:], energy[1:], color=color, linewidth=.8, alpha=.75)
        counts_ax.bar(np.arange(2) + (i - (len(labels)-1)/2) * .8/len(labels),
                      counts, width=.76/len(labels), color=color)
    corr_ax.set(title="АКФ", xlabel="Лаг, такты", ylabel="R[k]")
    corr_ax.legend(ncol=2, fontsize=8)
    dc = [values[2][0] for values in stats]
    energy_ax.set(title=f"Энергетический спектр · E[0]: {min(dc):.1f}–{max(dc):.1f}",
                  xlabel="Частота, циклов/такт", ylabel="Энергия")
    counts_ax.set(title="Гистограмма выхода", xlabel="Значение y", ylabel="Доля", xticks=[0, 1])
    distances = np.count_nonzero(runs != runs[0], axis=1)
    dist_ax.bar(labels, distances, color=[f"C{i}" for i in range(len(labels))])
    dist_ax.set(title="Отличающиеся отсчёты от старта z1", xlabel="Начальное состояние",
                ylabel="Количество", ylim=(0, max(1, distances.max() * 1.2)))
    if len(labels) > 6:
        dist_ax.tick_params(axis="x", labelrotation=35)
    fig.suptitle(title)
    return fig, stats


def markov_plot(matrix, final, frequencies, chain):
    fig = figure()
    matrix_ax, freq_ax, trace_ax, limit_ax = fig.subplots(2, 2).flat
    image = matrix_ax.imshow(matrix, cmap="Blues", origin="upper")
    fig.colorbar(image, ax=matrix_ax, shrink=.82)
    ticks = np.arange(10)
    matrix_ax.set(title="Вероятности перехода", xlabel="Новое состояние", ylabel="Прежнее состояние",
                  xticks=ticks, xticklabels=ticks+1, yticks=ticks, yticklabels=ticks+1)
    for row in frequencies:
        freq_ax.plot(ticks+1, row, color="#2563eb", alpha=.25)
    freq_ax.bar(ticks+1, frequencies[0], alpha=.3, color="#2563eb", label="Частоты из z1")
    freq_ax.plot(ticks+1, final, "o-", color="#d97706", label="Теория")
    freq_ax.set(title="Гистограмма состояний · 10 стартов", xlabel="Состояние", ylabel="Доля",
                xticks=ticks+1)
    freq_ax.legend(fontsize=8)
    trace_ax.step(np.arange(1, 81), chain[:80]+1, where="post", color="#0f766e")
    trace_ax.set(title="Первые 80 тактов из z1", xlabel="Такт", ylabel="Состояние")
    distribution = np.eye(10)
    deviations = []
    for step in range(41):
        deviations.append(np.abs(distribution - final).sum(axis=1).max()/2)
        distribution = distribution @ matrix
    limit_ax.semilogy(range(41), deviations, color="#0f766e")
    limit_ax.set(title="Сходимость распределений из всех стартов", xlabel="Такт",
                 ylabel="Макс. расстояние полной вариации")
    fig.suptitle("Марковская цепь на десяти состояниях")
    return fig, deviations[-1]


def save_tables(matrix, stochastic, chain, final, frequencies, chain_runs, batches):
    def save(name, header, rows):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(header)
            writer.writerows(rows)

    save("codes.csv", ("Буква", "Код"), ((c, f"{i:05b}") for i, c in enumerate(ALPHABET)))
    save("moore_matrix.csv", ("Состояние", "Вход 0", "Вход 1", "Вход 2", "Выход"),
         ((i+1, *(row+1), MOORE_Y[i]) for i, row in enumerate(matrix)))
    for name, transition in (("probability_input_0", stochastic[0]),
                             ("probability_input_1", stochastic[1]), ("markov_matrix", chain)):
        save(f"{name}.csv", ("Состояние", *range(1, 11)),
             ((i+1, *[f"{p:.6f}" for p in row]) for i, row in enumerate(transition)))
    save("markov_frequencies.csv", ("Состояние", "Теория", *[f"Старт z{i}" for i in range(1, 11)]),
         ((i+1, f"{final[i]:.6f}", *[f"{p:.6f}" for p in frequencies[:, i]]) for i in range(10)))
    save("markov_states.csv", ("Такт", *[f"Старт z{i}" for i in range(1, 11)]),
         ((t+1, *(row+1)) for t, row in enumerate(chain_runs.T)))
    for name, starts, signal, runs, states, stats in batches:
        save(f"{name}_summary.csv", ("Старт", "P(y=0)", "P(y=1)", "R[0]", "E[0]", "Отличий от z1"),
             ((i+1, *counts, corr[0], power[0], np.count_nonzero(run != runs[0]))
              for i, run, (corr, _, power, counts) in zip(starts, runs, stats)))
        save(f"{name}_signals.csv", ("Такт", "Вход", *[item for i in starts
             for item in (f"Состояние из z{i+1}", f"Выход из z{i+1}")]),
             ((t+1, symbol, *[item for trajectory, run in zip(states, runs)
              for item in (trajectory[t]+1, run[t])]) for t, symbol in enumerate(signal)))
        labels = [f"Старт z{i+1}" for i in starts]
        save(f"{name}_acf.csv", ("Лаг", *labels),
             ((k, *[corr[k] for corr, *_ in stats]) for k in range(len(stats[0][0]))))
        save(f"{name}_spectrum.csv", ("Частота", *labels),
             ((f, *[power[k] for _, _, power, _ in stats]) for k, f in enumerate(stats[0][1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="только создать PNG и CSV")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    moore_input = rng.integers(0, 3, STEPS)
    random_input = rng.integers(0, 2, STEPS)
    uniforms = rng.random(STEPS)  # Одинаковая случайность помогает изолировать эффект старта.

    matrix = moore_matrix()
    transitions = random_matrices()
    moore_runs = [moore(moore_input, i, matrix) for i in MOORE_STARTS]
    moore_states, moore_outputs = (np.stack(values) for values in zip(*moore_runs))
    random_states = np.stack([random_trajectory(random_input, i, transitions, uniforms)
                              for i in range(10)])
    random_outputs = RANDOM_Y[random_states]
    a, a_stats = comparison(moore_outputs, [f"z{i+1}" for i in MOORE_STARTS], "Автомат Мура")
    b, b_stats = comparison(random_outputs, [f"z{i+1}" for i in range(10)], "Y-детерминированный автомат")
    p = markov_matrix()
    final = stationary(p)
    # Отдельная реализация из каждого старта, по 1000 переходов.
    chain_runs = np.stack([random_trajectory(np.zeros(STEPS, dtype=int), i, p[None],
                                             rng.random(STEPS)) for i in range(10)])
    frequencies = np.stack([np.bincount(run, minlength=10)/STEPS for run in chain_runs])
    markov, distance = markov_plot(p, final, frequencies, chain_runs[0])
    print("Финальные вероятности:", np.round(final, 6))
    print(f"Максимальное расстояние до них на 40-м шаге: {distance:.6g}")

    pages = [
        ("01-codes", "Шифратор", "Каждая из 32 букв получает пятибитный код. "
         "Дешифратор по коду восстанавливает исходную букву.", coding_plot()),
        ("02-moore-graph", "Граф Мура", "В узле указаны состояние и его выход. "
         "Цвет стрелки показывает входной символ, который вызывает переход.", moore_graph()),
        ("03-moore-trace", "Такты Мура", "Первые 80 из 1000 тактов при старте из z1. "
         "Выход считывается из состояния после перехода.",
         snapshot(moore_input, moore_states[0], moore_outputs[0], "Автомат Мура · старт z1")),
        ("04-moore-analysis", "Анализ Мура", "Четыре старта при одинаковом входе. "
         "В спектре E[0] подписана отдельно, чтобы видеть остальные частоты.", a),
        ("05-random-trace", "Случайные такты", "Первые 80 тактов случайных переходов при старте из z1. "
         "Выход однозначно задаётся достигнутым состоянием.",
         snapshot(random_input, random_states[0], random_outputs[0], "Y-детерминированный автомат · старт z1")),
        ("06-random-analysis", "Анализ переходов", "Десять стартов на общем входе и общей последовательности "
         "случайных чисел. Справа внизу показано число несовпадений с выходом из z1.", b),
        ("07-markov", "Цепь Маркова", "Слева матрица и первые такты, справа сравнение частот "
         "с теорией и сходимость распределений из всех стартов.", markov),
    ]
    save_tables(matrix, transitions, p, final, frequencies, chain_runs, [
        ("moore", MOORE_STARTS, moore_input, moore_outputs, moore_states, a_stats),
        ("probabilistic", range(10), random_input, random_outputs, random_states, b_stats),
    ])
    for name, _, _, fig in pages:
        fig.savefig(OUT / f"{name}.png", dpi=135)
    print(f"Графики и таблицы: {OUT}")
    if not args.no_gui:
        show_gallery("ЛР № 6 · Конечные автоматы", pages)


if __name__ == "__main__":
    main()
