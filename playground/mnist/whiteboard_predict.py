import os
import tkinter as tk
from tkinter import messagebox

import numpy as np

from utils_mnist import ClassificationNetwork, build_mnist_layers

GRID_SIZE = 28
SCALE = 10
CANVAS_SIZE = GRID_SIZE * SCALE


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


class WhiteboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PiTensor MNIST Whiteboard")

        self.pixels = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.rects = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self.approach = tk.StringVar(value="mlp")
        self.status = tk.StringVar(value="Model: not loaded")
        self.prediction = tk.StringVar(value="Prediction: -")
        self.brush_radius = tk.IntVar(value=2)
        self.brush_strength = tk.DoubleVar(value=0.7)

        self.net = None

        self._build_ui()
        self._load_model()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        canvas_frame = tk.Frame(frame)
        canvas_frame.grid(row=0, column=0, rowspan=6, padx=(0, 10))

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#444444",
        )
        self.canvas.pack()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x0 = c * SCALE
                y0 = r * SCALE
                rect = self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + SCALE,
                    y0 + SCALE,
                    fill="#000000",
                    outline="",
                )
                self.rects[r][c] = rect

        self.canvas.bind("<Button-1>", self._on_paint)
        self.canvas.bind("<B1-Motion>", self._on_paint)

        controls = tk.Frame(frame)
        controls.grid(row=0, column=1, sticky="nw")

        tk.Label(controls, text="Model").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(
            controls,
            text="MLP",
            variable=self.approach,
            value="mlp",
            command=self._load_model,
        ).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(
            controls,
            text="CNN",
            variable=self.approach,
            value="cnn",
            command=self._load_model,
        ).grid(row=2, column=0, sticky="w")

        tk.Label(controls, textvariable=self.status, fg="#444444").grid(row=3, column=0, sticky="w", pady=(6, 0))
        tk.Label(controls, textvariable=self.prediction, font=("Arial", 12, "bold")).grid(row=4, column=0, sticky="w", pady=(6, 0))

        tk.Label(controls, text="Brush radius").grid(row=5, column=0, sticky="w", pady=(10, 0))
        tk.Scale(
            controls,
            from_=1,
            to=4,
            orient="horizontal",
            variable=self.brush_radius,
            length=160,
        ).grid(row=6, column=0, sticky="w")

        tk.Label(controls, text="Brush strength").grid(row=7, column=0, sticky="w")
        tk.Scale(
            controls,
            from_=0.2,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.brush_strength,
            length=160,
        ).grid(row=8, column=0, sticky="w")

        tk.Button(controls, text="Predict", command=self._predict).grid(row=9, column=0, sticky="w", pady=(10, 0))
        tk.Button(controls, text="Clear", command=self._clear).grid(row=10, column=0, sticky="w", pady=(4, 0))

    def _build_model(self) -> ClassificationNetwork:
        layers = build_mnist_layers(self.approach.get())
        return ClassificationNetwork(layers)

    def _load_model(self) -> None:
        approach = self.approach.get()
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        checkpoint_path = os.path.join(root_dir, "playground", "checkpoints", f"{approach}_mnist", "best.npy")
        if not os.path.exists(checkpoint_path):
            self.net = None
            self.status.set(f"Model: missing {approach}_mnist/best.npy")
            return

        self.net = self._build_model()
        self.net.load_parameters(checkpoint_path)
        self.status.set(f"Model: loaded {approach}")

    def _on_paint(self, event: tk.Event) -> None:
        x = event.x // SCALE
        y = event.y // SCALE
        self._apply_brush(x, y)

    def _apply_brush(self, x: int, y: int) -> None:
        radius = self.brush_radius.get()
        strength = self.brush_strength.get()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                row = y + dy
                col = x + dx
                if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
                    continue
                dist = (dx * dx + dy * dy) ** 0.5
                weight = 1.0 - (dist / (radius + 1e-6))
                value = self.pixels[row, col]
                new_value = min(1.0, value + strength * weight)
                if new_value != value:
                    self.pixels[row, col] = new_value
                    self._update_cell(row, col)

    def _update_cell(self, row: int, col: int) -> None:
        intensity = int(self.pixels[row, col] * 255)
        color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
        self.canvas.itemconfig(self.rects[row][col], fill=color)

    def _clear(self) -> None:
        self.pixels.fill(0.0)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._update_cell(r, c)
        self.prediction.set("Prediction: -")

    def _predict(self) -> None:
        if self.net is None:
            self._load_model()
            if self.net is None:
                messagebox.showerror("Model not found", "Train a model first or add best.npy checkpoints.")
                return

        img = self.pixels.copy()
        if self.approach.get() == "mlp":
            input_data = img.reshape(1, -1)
        else:
            input_data = img.reshape(1, 1, GRID_SIZE, GRID_SIZE)

        logits = self.net.layers.forward(input_data)
        pred = int(np.argmax(logits, axis=1)[0])
        probs = softmax(logits[0])
        top3 = np.argsort(-probs)[:3]
        top3_str = ", ".join([f"{i}:{probs[i]:.2f}" for i in top3])
        self.prediction.set(f"Prediction: {pred} (top-3: {top3_str})")


def main() -> None:
    root = tk.Tk()
    WhiteboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
