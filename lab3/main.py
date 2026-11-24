"""ЛР № 7. Одноканальная СМО с конечным буфером M/M/1/(B+1)."""
import argparse
from collections import deque
import csv
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure

from viewer import show_gallery

OUT = Path(__file__).resolve().parent / "results"
COLUMNS = ("Отказы, шт.", "Занятость", "Время в СМО, с", "Длина очереди",
           "Задержка, с", "Доля отказов", "Пропускная способность, заявок/с")


def flow(n, rate, rng, kind="poisson", shape=3):
    """Возвращает интервалы и моменты событий при среднем интервале 1/rate."""
    if kind == "poisson":
        intervals = rng.exponential(1 / rate, n)
    elif kind == "erlang":
        # Сумма shape независимых экспонент с интенсивностью shape*rate.
        intervals = rng.gamma(shape, 1 / (rate * shape), n)
    elif kind == "weibull":
        from math import gamma
        intervals = rng.weibull(shape, n) / (rate * gamma(1 + 1 / shape))
    else:
        raise ValueError(f"Неизвестный поток: {kind}")
    return intervals, intervals.cumsum()


def simulate(n, rate, service_rate, buffer, rng):
    """Обрабатывает приходы и завершения в порядке модельного времени."""
    intervals, arrivals = flow(n, rate, rng)
    service_times = rng.exponential(1 / service_rate, n)
    ends = deque()
    accepted, rejected, departures = [], [], []
    waits, services, visits = [], [], []
    # В календаре сохраняем фактическое число заявок после каждого события.
    clock, states = [0.0], [0]
    horizon = arrivals[-1]

    for arrival, duration in zip(arrivals, service_times):
        # Если завершение совпало с приходом, сначала освобождаем место.
        while ends and ends[0] <= arrival:
            finish = ends.popleft()
            departures.append(finish)
            clock.append(finish)
            states.append(len(ends))

        # В системе помещаются B ожидающих и одна заявка в канале.
        if len(ends) == buffer + 1:
            rejected.append(arrival)
            continue

        # Последняя заявка в календаре определяет момент начала новой.
        start = ends[-1] if ends else arrival
        finish = start + duration
        ends.append(finish)
        accepted.append((arrival, start, finish))
        clock.append(arrival)
        states.append(len(ends))
        waits.append(start - arrival)
        visits.append(finish - arrival)
        services.append(duration)

    # Занятость и очередь считаем на [0, t_N]. Для каждой принятой
    # заявки индивидуальные времена известны и после конца наблюдения.
    widths = np.diff([*clock, horizon])
    levels = np.asarray(states)
    busy = widths[levels > 0].sum() / horizon
    queue = widths @ np.maximum(levels - 1, 0) / horizon
    metrics = np.array([len(rejected), busy, np.mean(visits), queue,
                        np.mean(waits), len(rejected) / n, len(departures) / horizon])
    return metrics, (intervals, services, accepted, rejected, horizon, clock, levels)


def theoretical(n, rate, service_rate, buffer):
    """Стационарные показатели конечной цепи рождения и гибели."""
    states = np.arange(buffer + 2)
    # Вычитание максимума оставляет те же вероятности и защищает от overflow.
    log_weights = states * (np.log(rate) - np.log(service_rate))
    weights = np.exp(log_weights - log_weights.max())
    p = weights / weights.sum()
    refusals = p[-1]
    busy = p[1:].sum()
    queue = p @ np.maximum(states - 1, 0)
    throughput = service_rate * busy
    # Закон Литтла для принятых: делим на эффективную интенсивность A.
    return np.array([n * refusals, busy, (queue + busy) / throughput,
                     queue, queue / throughput, refusals, throughput])


def demo_plot(figure, result, rate, service_rate, buffer, n):
    figure.clear()
    _, (intervals, services, accepted, rejected, horizon, clock, levels) = result
    axes = figure.subplots(2, 2)
    timeline, occupancy, entry, service = axes.flat
    arrivals = np.cumsum(intervals)
    timeline.scatter(arrivals, np.full(n, 2), s=18, color="#2563eb", label="Приход")
    timeline.scatter(rejected, np.full(len(rejected), 2), s=32, marker="x",
                     color="#dc2626", label="Отказ")
    for arrival, start, finish in accepted:
        if start > arrival:
            timeline.hlines(1, arrival, start, color="#d97706", linewidth=3)
        timeline.hlines(0, start, finish, color="#0f766e", linewidth=3)
    timeline.set(yticks=[0, 1, 2], yticklabels=["Канал", "Буфер", "Вход"],
                 xlabel="Время, с", title="Путь заявок")
    timeline.legend(loc="lower right", fontsize=8)
    occupancy.step([*clock, horizon], [*levels, levels[-1]], where="post", color="#2563eb", label="Всего")
    occupancy.step([*clock, horizon], [*np.maximum(levels - 1, 0), max(levels[-1]-1, 0)],
                   where="post", color="#d97706", label="В очереди")
    occupancy.set(xlabel="Время, с", ylabel="Заявок", title="В системе и в очереди")
    occupancy.legend()
    for ax, values, title in [(entry, intervals, "Интервалы между приходами"),
                               (service, services, "Длительности обслуживания")]:
        ax.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))), color="#0f766e")
        ax.set(xlabel="Интервал, с", ylabel="Количество", title=title)
    figure.suptitle(f"СМО при λ = {rate:g}, μ = {service_rate:g}, B = {buffer}, N = {n}")


def flow_plot():
    fig = Figure(figsize=(10, 6), layout="constrained")
    rng = np.random.default_rng(64)
    for ax, kind, title in zip(fig.subplots(1, 3), ("poisson", "erlang", "weibull"),
                               ("Пуассон", "Эрланг (k = 3)", "Вейбулл (k = 3)")):
        for rate in (1, 2):
            intervals, _ = flow(3000, rate, rng, kind)
            ax.hist(intervals, bins=np.linspace(0, 5, 45), density=True, histtype="step",
                    linewidth=1.7, label=f"λ = {rate}")
        ax.set(title=title, xlabel="Интервал, с", ylabel="Плотность")
        ax.legend()
    return fig


def algorithms_plot():
    fig = Figure(figsize=(10, 6), layout="constrained")
    for ax, title, finite in zip(fig.subplots(1, 2),
                                 ("С ожиданием (очередь не ограничена)",
                                  "С конечным буфером и отказами"), (False, True)):
        ax.set(title=title, xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        nodes = [
            (.5, .91, "Начало: канал свободен, FIFO пуста"),
            (.5, .74, "Выбрать ближайшее событие"),
            (.25, .54, "Приход заявки"),
            (.75, .54, "Завершение обслуживания"),
            (.275, .34, ("Канал свободен → начать\nИначе → в FIFO, если есть место\nБуфер полон → отказ" if finite
                         else "Канал свободен → начать\nКанал занят → в FIFO")),
            (.75, .34, "FIFO пуста → освободить канал\nИначе → обслужить первую"),
            (.5, .11, "Обновить статистику"),
        ]
        for x, y, text in nodes:
            ax.text(x, y, text, ha="center", va="center", fontsize=9, zorder=3,
                    bbox=dict(boxstyle="round,pad=.45", fc="#e0f2fe", ec="#0369a1"))
        for origin, target in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)]:
            x, y = nodes[origin][:2]
            start = (x, y - .036) if origin in (4, 5) else (x, y)
            ax.annotate("", nodes[target][:2], start,
                        arrowprops=dict(arrowstyle="->", shrinkA=12, shrinkB=18))
        ax.plot([.34, .018, .018, .25], [.11, .11, .74, .74], color="#0f766e", lw=1.1)
        ax.annotate("", (.29, .74), (.24, .74),
                    arrowprops=dict(arrowstyle="->", color="#0f766e"))
    return fig


def grid_experiment(rate, rng):
    """Для каждой комбинации B и μ усредняем 5 независимых прогонов по 200 заявок."""
    empirical = np.empty((5, 5, len(COLUMNS)))
    theory = np.empty_like(empirical)
    for b in range(1, 6):
        for mu in range(1, 6):
            empirical[b-1, mu-1] = np.mean(
                [simulate(200, rate, mu, b, rng)[0] for _ in range(5)], axis=0)
            theory[b-1, mu-1] = theoretical(200, rate, mu, b)
    return empirical, theory


def surface_plot(rate, empirical, theory, metric):
    fig = Figure(figsize=(10, 5.8), layout="constrained")
    b, mu = np.meshgrid(np.arange(1, 6), np.arange(1, 6), indexing="ij")
    limit = max(empirical[..., metric].max(), theory[..., metric].max())
    for pos, values in enumerate((empirical, theory), start=1):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.plot_surface(b, mu, values[..., metric], cmap="viridis", vmin=0, vmax=limit)
        ax.set(title=("Моделирование" if pos == 1 else "Теория"),
               xlabel="Буфер B", ylabel="Интенсивность μ", zlim=(0, limit * 1.06),
               xticks=range(1, 6), yticks=range(1, 6))
    fig.suptitle(f"{COLUMNS[metric]} при λ = {rate:g}")
    return fig


def save_table(experiments):
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(("λ", "μ", "B", "Метод", *COLUMNS))
        for rate, empirical, theory in experiments:
            for b in range(1, 6):
                for mu in range(1, 6):
                    for method, matrix in (("Опыт", empirical), ("Теория", theory)):
                        writer.writerow((rate, mu, b, method,
                                         *[f"{v:.6f}".replace(".", ",") for v in matrix[b-1, mu-1]]))


def controls_for(figure):
    rng = np.random.default_rng(177)

    def add(panel):
        entries = []
        for i, (name, default) in enumerate((("λ", 1), ("μ", 1), ("Буфер B", 1), ("Заявок N", 20))):
            ttk.Label(panel, text=name).grid(row=0, column=2*i, padx=(4, 3))
            field = ttk.Entry(panel, width=7)
            field.insert(0, str(default))
            field.grid(row=0, column=2*i+1, padx=(0, 7))
            entries.append(field)
        status = ttk.Label(panel, text="Начальный опыт: 20 заявок")
        status.grid(row=1, column=0, columnspan=9, sticky="w", pady=(5, 0))

        def run():
            try:
                rate, mu = (float(field.get().replace(",", ".")) for field in entries[:2])
                b, n = (int(field.get()) for field in entries[2:])
                if not (1e-6 <= rate <= 1e6 and 1e-6 <= mu <= 1e6 and 0 <= b <= 1000 and 1 <= n <= 5000):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Параметры", "λ и μ: от 0,000001 до 1 000 000.\n"
                                     "B: целое 0–1000. N: целое 1–5000.")
                return
            result = simulate(n, rate, mu, b, rng)
            demo_plot(figure, result, rate, mu, b, n)
            status.configure(text=f"Отказы: {result[0][0]:.0f} из {n}  •  "
                                  f"Занятость канала: {result[0][1]:.1%}")
            panel.redraw()

        ttk.Button(panel, text="Пуск", command=run).grid(row=0, column=8, padx=(4, 0))

    return add


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="только создать PNG и CSV")
    parser.add_argument("--extra-rate", type=float, default=2.0, help="второе значение λ")
    args = parser.parse_args()
    if not np.isfinite(args.extra_rate) or args.extra_rate <= 0 or args.extra_rate == 1:
        parser.error("--extra-rate должно быть конечным положительным числом, отличным от 1")
    OUT.mkdir(exist_ok=True)

    # Четыре прогона для просмотра. В отчёт берём второй фиксированный пример.
    rng = np.random.default_rng(70)
    demos = [simulate(20, 1, 1, 1, rng) for _ in range(4)]
    demo = Figure(figsize=(10, 6), layout="constrained")
    demo_plot(demo, demos[1], 1, 1, 1, 20)
    pages = [
        ("01-experiment", "Прогон", "Приходы и отказы показаны точками и крестиками. "
         "Отрезки на дорожках отмечают ожидание в буфере и работу канала.", demo),
        ("02-flows", "Потоки", "Плотности интервалов для трёх потоков. При большей интенсивности "
         "λ средний интервал сокращается.", flow_plot()),
        ("03-algorithms", "Алгоритмы", "Две схемы обслуживания. FIFO означает, что из очереди "
         "первой уходит заявка, пришедшая раньше остальных.", algorithms_plot()),
    ]
    experiments = []
    rng = np.random.default_rng(2026)
    for i, rate in enumerate((1, args.extra_rate), start=1):
        empirical, theory = grid_experiment(rate, rng)
        experiments.append((rate, empirical, theory))
        for metric, name, description in (
            (0, "Отказы", "Число отклонённых заявок из 200. Каждая точка опыта усреднена по пяти прогонам."),
            (1, "Занятость", "Доля времени, когда канал обслуживал заявку. Оси основания — размер буфера B и μ."),
        ):
            pages.append((f"0{i*2+metric+2}-{['refusals', 'busy'][metric]}-{i}",
                          f"{name} λ={rate:g}", description, surface_plot(rate, empirical, theory, metric)))
    save_table(experiments)
    for filename, _, _, fig in pages:
        fig.savefig(OUT / f"{filename}.png", dpi=135)
    print(f"Графики и таблица: {OUT}")
    if not args.no_gui:
        show_gallery("ЛР № 7 · Одноканальная СМО", pages, controls_for(demo))


if __name__ == "__main__":
    main()
