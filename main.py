import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LeastSquaresApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Аппроксимация МНК")
        self.root.geometry("1200x800")
        self.data = []
        self.models = [
            ('Линейная', self.fit_linear, self.linear, 2),
            ('Квадратичная', self.fit_quadratic, self.quadratic, 3),
            ('Кубическая', self.fit_cubic, self.cubic, 4),
            ('Экспоненциальная', self.fit_exponential, self.exponential, 2),
            ('Логарифмическая', self.fit_logarithmic, self.logarithmic, 2),
            ('Степенная', self.fit_power, self.power, 2)
        ]
        self.init_ui()

    def init_ui(self):
        # Фрейм управления
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Ручной ввод данных
        ttk.Label(control_frame, text="Ручной ввод:").pack(pady=5)
        self.x_entry = ttk.Entry(control_frame, width=10)
        self.x_entry.pack()
        self.y_entry = ttk.Entry(control_frame, width=10)
        self.y_entry.pack()
        ttk.Button(control_frame, text="Добавить точку", command=self.add_point).pack(pady=5)

        # Список точек
        ttk.Label(control_frame, text="Точки:").pack(pady=(20,5))
        self.points_listbox = tk.Listbox(control_frame, height=10)
        self.points_listbox.pack(fill=tk.X)

        # Управление данными
        ttk.Button(control_frame, text="Загрузить файл", command=self.load_file).pack(pady=5)
        ttk.Button(control_frame, text="Очистить данные", command=self.clear_data).pack(pady=5)
        ttk.Button(control_frame, text="Рассчитать", command=self.calculate).pack(pady=5)

        # Вывод результатов
        self.result_text = tk.Text(self.root, wrap=tk.WORD)
        self.result_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # График
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_point(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            self.data.append((x, y))
            self.x_entry.delete(0, tk.END)
            self.y_entry.delete(0, tk.END)
            # Обновляем список и график
            self.update_points_list()
            self.redraw_points()
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные данные")

    def update_points_list(self):
        self.points_listbox.delete(0, tk.END)
        for x, y in self.data:
            self.points_listbox.insert(tk.END, f"({x:.3f}, {y:.3f})")

    def redraw_points(self):
        self.ax.clear()
        if self.data:
            x_vals, y_vals = zip(*self.data)
            self.ax.scatter(x_vals, y_vals, label='Данные')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.data = [tuple(map(float, line.strip().split())) for line in f]
                messagebox.showinfo("Успех", f"Загружено {len(self.data)} точек")
                self.update_points_list()
                self.redraw_points()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def clear_data(self):
        self.data = []
        self.points_listbox.delete(0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.result_text.delete(1.0, tk.END)

    def gauss_elimination(self, matrix):
        n = len(matrix)
        for i in range(n):
            max_row = i
            for r in range(i, n):
                if abs(matrix[r][i]) > abs(matrix[max_row][i]):
                    max_row = r
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

            pivot = matrix[i][i]
            if abs(pivot) < 1e-10:
                return None

            for j in range(i + 1, n):
                factor = matrix[j][i] / pivot
                for k in range(i, n + 1):
                    matrix[j][k] -= factor * matrix[i][k]

        solution = [0] * n
        for i in range(n - 1, -1, -1):
            sum_ = 0.0
            for j in range(i + 1, n):
                sum_ += matrix[i][j] * solution[j]
            solution[i] = (matrix[i][-1] - sum_) / matrix[i][i]

        return solution

    # Функции моделей для рисования
    def linear(self, x, a, b):
        return a * x + b

    def quadratic(self, x, a, b, c):
        return a * x ** 2 + b * x + c

    def cubic(self, x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    def exponential(self, x, a, b):
        return a * math.exp(b * x)

    def logarithmic(self, x, a, b):
        return a * math.log(x) + b

    def power(self, x, a, b):
        return a * (x ** b)

    def linear_regression(self, X, Y):
        n = len(X)

        sum_x = 0.0
        for xi in X:
            sum_x += xi

        sum_y = 0.0
        for yi in Y:
            sum_y += yi

        sum_xy = 0.0
        for i in range(n):
            sum_xy += X[i] * Y[i]

        sum_x2 = 0.0
        for xi in X:
            sum_x2 += xi ** 2

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y * sum_x2 - sum_x * sum_xy) / denom
        return [slope, intercept]

    def fit_linear(self):
        x = []
        y = []
        for p in self.data:
            x.append(p[0])
            y.append(p[1])

        return self.linear_regression(x, y)

    def fit_quadratic(self):
        x = []
        y = []
        for p in self.data:
            x.append(p[0])
            y.append(p[1])

        n = len(x)

        sum_x2_y = 0.0
        sum_x = 0.0
        sum_x2 = 0.0
        sum_x3 = 0.0
        sum_x4 = 0.0
        sum_y = 0.0
        sum_x_y = 0.0
        sum_x3_y = 0.0

        for i in range(n):
            xi = x[i]
            yi = y[i]
            xi2 = xi ** 2
            xi3 = xi2 * xi
            xi4 = xi3 * xi

            sum_x += xi
            sum_x2 += xi2
            sum_x3 += xi3
            sum_x4 += xi4
            sum_y += yi

            sum_x_y += xi * yi
            sum_x2_y += xi2 * yi
            sum_x3_y += xi3 * yi

        matrix = [
            [sum_x4, sum_x3, sum_x2, sum_x2_y],
            [sum_x3, sum_x2, sum_x, sum_x_y],
            [sum_x2, sum_x, n, sum_y]
        ]

        return self.gauss_elimination(matrix)

    def fit_cubic(self):
        x = []
        y = []
        for p in self.data:
            x.append(p[0])
            y.append(p[1])

        n = len(x)

        sum_x = sum_x2 = sum_x3 = sum_x4 = sum_x5 = sum_x6 = 0.0
        sum_y = 0.0
        sum_x_y = sum_x2_y = sum_x3_y = sum_x4_y = 0.0

        for i in range(n):
            xi = x[i]
            yi = y[i]
            xi2 = xi ** 2
            xi3 = xi2 * xi
            xi4 = xi3 * xi
            xi5 = xi4 * xi
            xi6 = xi5 * xi

            sum_x += xi
            sum_x2 += xi2
            sum_x3 += xi3
            sum_x4 += xi4
            sum_x5 += xi5
            sum_x6 += xi6
            sum_y += yi

            sum_x_y += xi * yi
            sum_x2_y += xi2 * yi
            sum_x3_y += xi3 * yi
            sum_x4_y += xi4 * yi

        matrix = [
            [sum_x6, sum_x5, sum_x4, sum_x3, sum_x3_y],
            [sum_x5, sum_x4, sum_x3, sum_x2, sum_x2_y],
            [sum_x4, sum_x3, sum_x2, sum_x, sum_x_y],
            [sum_x3, sum_x2, sum_x, n, sum_y]
        ]

        return self.gauss_elimination(matrix)

    def fit_exponential(self):
        x = []
        y = []
        for p in self.data:
            x.append(p[0])
            y.append(p[1])

        x_filt = []
        y_filt = []
        for i in range(len(x)):
            if y[i] > 0:
                x_filt.append(x[i])
                y_filt.append(y[i])

        if len(x_filt) < 2:
            return None

        Y_log = []
        for yi in y_filt:
            Y_log.append(math.log(yi))

        coeffs_lin = self.linear_regression(x_filt, Y_log)
        if not coeffs_lin:
            return None

        b = coeffs_lin[0]
        ln_a = coeffs_lin[1]
        a = math.exp(ln_a)
        return [a, b]

    def fit_logarithmic(self):
        x = []
        y = []
        for p in self.data:
            x.append(p[0])
            y.append(p[1])

        x_filt = []
        y_filt = []
        for i in range(len(x)):
            if x[i] > 0:
                x_filt.append(x[i])
                y_filt.append(y[i])

        if len(x_filt) < 2:
            return None

        X_log = []
        for xi in x_filt:
            X_log.append(math.log(xi))

        coeffs_lin = self.linear_regression(X_log, y_filt)
        if not coeffs_lin:
            return None

        a = coeffs_lin[0]
        b = coeffs_lin[1]
        return [a, b]

    def fit_power(self):
        x = []
        y = []
        for p in self.data:
            x.append(p[0])
            y.append(p[1])

        x_filt = []
        y_filt = []
        for i in range(len(x)):
            if x[i] > 0 and y[i] > 0:
                x_filt.append(x[i])
                y_filt.append(y[i])

        if len(x_filt) < 2:
            return None

        X_log = []
        Y_log = []
        for i in range(len(x_filt)):
            X_log.append(math.log(x_filt[i]))
            Y_log.append(math.log(y_filt[i]))

        coeffs_lin = self.linear_regression(X_log, Y_log)
        if not coeffs_lin:
            return None

        b = coeffs_lin[0]
        ln_a = coeffs_lin[1]
        a = math.exp(ln_a)
        return [a, b]

    def calculate(self):
        if len(self.data) < 4:
            messagebox.showerror("Ошибка", "Недостаточно данных (минимум 4 точки)")
            return

        results = []
        for name, fit_fn, func, _ in self.models:
            try:
                coeffs = fit_fn()
                if not coeffs:
                    continue

                # Расчёт ошибок
                sse = 0.0
                y_mean = sum(y for x, y in self.data) / len(self.data)
                sst = sum((y - y_mean) ** 2 for x, y in self.data)

                for x_val, y_val in self.data:
                    try:
                        y_pred = func(x_val, *coeffs)
                        sse += (y_val - y_pred) ** 2
                    except:
                        sse = float('inf')
                        break

                mse = math.sqrt(sse / len(self.data)) if sse != float('inf') else float('inf')
                r2 = 1 - (sse / sst) if sst != 0 else 0

                # Корреляцию Пирсона считаем только для линейной модели
                pearson = self.calculate_pearson() if name == 'Линейная' else None

                results.append({
                    'name': name,
                    'coeffs': coeffs,
                    'mse': mse,
                    'r2': r2,
                    'pearson': pearson,
                    'func': func
                })
            except Exception:
                continue

        if not results:
            messagebox.showerror("Ошибка", "Не удалось рассчитать ни одну модель")
            return

        # Вывод результатов
        self.result_text.delete(1.0, tk.END)
        best = min(results, key=lambda x: x['mse'])

        for res in results:
            text = f"{res['name']}:\n"
            text += f"  Коэффициенты: {', '.join(f'{c:.4f}' for c in res['coeffs'])}\n"
            text += f"  СКО (RMSE): {res['mse']:.4f}\n"
            text += f"  R²: {res['r2']:.4f}\n"
            if res['pearson'] is not None:
                text += f"  Корреляция Пирсона: {res['pearson']:.4f}\n"
            text += "\n"
            self.result_text.insert(tk.END, text)

        # Отрисовка графиков вместе с уже отрисованными точками
        self.ax.clear()
        x_vals = [p[0] for p in self.data]
        y_vals = [p[1] for p in self.data]
        self.ax.scatter(x_vals, y_vals, label='Данные')

        if best:
            x_min = min(x_vals)
            x_max = max(x_vals)
            x_plot = [x_min + (x_max - x_min) * i / 100 for i in range(101)]
            y_plot = [best['func'](x, *best['coeffs']) for x in x_plot]
            self.ax.plot(x_plot, y_plot, 'r-', label=f"Лучшая: {best['name']}")

        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def calculate_pearson(self):
        x = [p[0] for p in self.data]
        y = [p[1] for p in self.data]
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        return numerator / denominator if denominator != 0 else 0


if __name__ == "__main__":
    app = LeastSquaresApp()
    app.root.mainloop()