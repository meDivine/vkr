import numpy as np
import time
import csv
from skimage.color import rgb2lab, lab2rgb, deltaE_cie76
from scipy.interpolate import RegularGridInterpolator


def generate_lut(N, code_func, clip=True):
    """Генерация 3D LUT заданного размера"""
    grid = np.linspace(0, 1, N)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb_nodes = np.stack([r, g, b], -1).reshape(-1, 3)
    lab_nodes = rgb2lab(rgb_nodes.reshape(-1, 1, 3)).reshape(-1, 3)
    lab_nodes = code_func(lab_nodes.copy())

    if clip:
        lab_nodes[:, 0] = np.clip(lab_nodes[:, 0], 0, 100)
        lab_nodes[:, 1:] = np.clip(lab_nodes[:, 1:], -128, 127)

    rgb_out_nodes = lab2rgb(lab_nodes.reshape(-1, 1, 3)).reshape(-1, 3)
    lut = np.clip(rgb_out_nodes, 0, 1).reshape(N, N, N, 3)
    return lut, grid


def calculate_metrics(lut, grid, code_func, clip=True, n_samples=10000):
    """Расчет метрик качества LUT"""
    interp = RegularGridInterpolator((grid, grid, grid), lut)
    np.random.seed(42)
    test_rgb = np.random.rand(n_samples, 3)

    # Эталонное преобразование
    test_lab = rgb2lab(test_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    test_lab_transformed = code_func(test_lab.copy())
    if clip:
        test_lab_transformed[:, 0] = np.clip(test_lab_transformed[:, 0], 0, 100)
        test_lab_transformed[:, 1:] = np.clip(test_lab_transformed[:, 1:], -128, 127)

    # Преобразование через LUT
    test_rgb_lut = interp(test_rgb)
    test_lab_lut = rgb2lab(test_rgb_lut.reshape(-1, 1, 3)).reshape(-1, 3)

    # Метрики
    delta_e = deltaE_cie76(test_lab_transformed, test_lab_lut)
    test_rgb_truth = lab2rgb(test_lab_transformed.reshape(-1, 1, 3)).reshape(-1, 3)
    mse = np.mean((test_rgb_truth - test_rgb_lut) ** 2)
    psnr = float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

    return {
        'Mean Delta E': np.mean(delta_e), 'Max Delta E': np.max(delta_e),
        'PSNR (dB)': psnr, 'MSE': mse
    }


def run_tests():
    """Запуск серии тестов для разных размеров LUT"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ТОЧНОСТИ ГЕНЕРАЦИИ 3D LUT")
    print("=" * 70)

    test_formulas = [
        {'name': 'Увеличение яркости (L * 1.2)', 'func': lambda lab: lab * [1.2, 1.0, 1.0]},
        {'name': 'Инверсия яркости (100 - L)', 'func': lambda lab: lab * [1.0, 1.0, 1.0] - [100, 0, 0]},
        {'name': 'Насыщенность (a,b * 1.5)', 'func': lambda lab: lab * [1.0, 1.5, 1.5]},
        {'name': 'Комбинированное преобразование', 'func': lambda lab: lab * [1.1, 1.3, 1.3] - [5, 0, 0]}
    ]

    lut_sizes = [17, 33, 65]
    results = []

    for formula in test_formulas:
        print(f"\nФОРМУЛА: {formula['name']}")
        print(f"{'Размер LUT':<12} | {'Время (с)':<10} | {'Mean ΔE':<10} | {'Max ΔE':<10} | {'PSNR (dB)':<10}")
        print("-" * 70)

        formula_results = []
        for N in lut_sizes:
            start_time = time.time()
            lut, grid = generate_lut(N, formula['func'], clip=True)
            metrics = calculate_metrics(lut, grid, formula['func'], clip=True)
            elapsed_time = time.time() - start_time

            print(f"{N}³ ({N ** 3:>5}) | {elapsed_time:<10.3f} | {metrics['Mean Delta E']:<10.4f} | "
                  f"{metrics['Max Delta E']:<10.4f} | {metrics['PSNR (dB)']:<10.2f}")

            formula_results.append({'size': N, 'time': elapsed_time, 'mean_de': metrics['Mean Delta E'],
                                    'max_de': metrics['Max Delta E'], 'psnr': metrics['PSNR (dB)']})
        results.append({'name': formula['name'], 'data': formula_results})

    # Сводная таблица
    print(f"\n{'=' * 70}\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (для LUT 33³)\n{'=' * 70}")
    print(f"{'Формула':<40} | {'Mean ΔE':<10} | {'Max ΔE':<10} | {'PSNR (dB)':<10}")
    print("-" * 70)
    for result in results:
        data_33 = [d for d in result['data'] if d['size'] == 33][0]
        print(
            f"{result['name']:<40} | {data_33['mean_de']:<10.4f} | {data_33['max_de']:<10.4f} | {data_33['psnr']:<10.2f}")
    print("=" * 70)
    return results


def test_edge_cases():
    """Тестирование граничных случаев"""
    print("\n" + "=" * 70 + "\nТЕСТИРОВАНИЕ ГРАНИЧНЫХ СЛУЧАЕВ\n" + "=" * 70)

    edge_colors = {
        'Чёрный': np.array([[0.0, 0.0, 0.0]]), 'Белый': np.array([[1.0, 1.0, 1.0]]),
        'Красный': np.array([[1.0, 0.0, 0.0]]), 'Зелёный': np.array([[0.0, 1.0, 0.0]]),
        'Синий': np.array([[0.0, 0.0, 1.0]]), 'Серый 50%': np.array([[0.5, 0.5, 0.5]]),
    }

    formula = lambda lab: lab * [1.2, 1.0, 1.0]
    lut, grid = generate_lut(33, formula, clip=True)
    interp = RegularGridInterpolator((grid, grid, grid), lut)

    print(f"\n{'Цвет':<15} | {'RGB вход':<20} | {'RGB выход':<20} | {'ΔE':<10}\n" + "-" * 70)
    for name, rgb in edge_colors.items():
        lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        lab_transformed = np.clip(formula(lab.copy()) * [1, 1, 1] - [0, 0, 0], [0, -128, -128], [100, 127, 127])
        rgb_lut = interp(rgb)
        lab_lut = rgb2lab(rgb_lut.reshape(-1, 1, 3)).reshape(-1, 3)
        delta_e = deltaE_cie76(lab_transformed, lab_lut)[0]
        print(f"{name:<15} | {str(rgb[0]):<20} | {str(rgb_lut[0]):<20} | {delta_e:<10.4f}")
    print("=" * 70)


def export_results_to_csv(results, filename='lut_test_results.csv'):
    """Экспорт результатов в CSV для Excel"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Формула', 'Размер LUT', 'Время (с)', 'Mean Delta E', 'Max Delta E', 'PSNR (dB)'])
        for result in results:
            for data in result['data']:
                writer.writerow([result['name'], data['size'], f"{data['time']:.3f}",
                                 f"{data['mean_de']:.4f}", f"{data['max_de']:.4f}", f"{data['psnr']:.2f}"])
    print(f"\n✓ Результаты экспортированы в {filename}")


if __name__ == "__main__":
    results = run_tests()
    test_edge_cases()
    export_results_to_csv(results)
    print("\n✓ Тестирование завершено успешно!")