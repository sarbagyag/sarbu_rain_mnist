#!/usr/bin/env python3
# draw.py — draw a digit, Rain MLP predicts it
# Requirements: numpy (pip install numpy)
# Run: python3 draw.py

import tkinter as tk
import numpy as np
import struct
import os

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
CANVAS_SIZE = 420   # display size in pixels
GRID        = 28    # 28x28 MNIST grid
CELL        = CANVAS_SIZE // GRID

# ── Load weights from Rain binary format (int32 rows, int32 cols, float64 data)
def load(name):
    path = os.path.join(WEIGHTS_DIR, name)
    with open(path, "rb") as f:
        rows = struct.unpack("<i", f.read(4))[0]
        cols = struct.unpack("<i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float64)
    return data.reshape(rows, cols)

W1 = load("w1.bin")
b1 = load("b1.bin")
W2 = load("w2.bin")
b2 = load("b2.bin")

def forward(pixels):
    x  = pixels.reshape(-1, 1) / 255.0
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)
    z2 = W2 @ a1 + b2
    e  = np.exp(z2 - z2.max())
    return (e / e.sum()).flatten()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Rain MNIST")
        root.configure(bg="#1e1e1e")
        root.resizable(False, False)

        # ── Drawing canvas ──────────────────────────────────────────────
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair", highlightthickness=0
        )
        self.canvas.pack(padx=16, pady=(16, 6))
        self.canvas.bind("<B1-Motion>",        self.on_drag)
        self.canvas.bind("<ButtonPress-1>",    self.on_drag)
        self.canvas.bind("<ButtonRelease-1>",  self.on_release)

        self.pixels = np.zeros((GRID, GRID), dtype=np.float32)

        # ── Prediction label ────────────────────────────────────────────
        self.pred_var = tk.StringVar(value="Draw a digit above")
        pred_lbl = tk.Label(
            root, textvariable=self.pred_var,
            font=("Helvetica", 28, "bold"),
            fg="white", bg="#1e1e1e"
        )
        pred_lbl.pack(pady=(0, 6))

        # ── Confidence bars for digits 0-9 ─────────────────────────────
        bar_frame = tk.Frame(root, bg="#1e1e1e")
        bar_frame.pack(fill="x", padx=16, pady=(0, 6))

        self.bar_cvs = []
        BAR_W = CANVAS_SIZE - 30
        for i in range(10):
            row = tk.Frame(bar_frame, bg="#1e1e1e")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=str(i), width=2,
                     fg="#aaa", bg="#1e1e1e",
                     font=("Helvetica", 11)).pack(side="left")
            bc = tk.Canvas(row, width=BAR_W, height=18,
                           bg="#2a2a2a", highlightthickness=0)
            bc.pack(side="left")
            self.bar_cvs.append(bc)

        # ── Clear button ────────────────────────────────────────────────
        tk.Button(
            root, text="Clear", command=self.clear,
            font=("Helvetica", 13), bg="#333", fg="white",
            activebackground="#555", relief="flat",
            padx=20, pady=6
        ).pack(pady=(4, 16))

        self.last_probs = None

    # ── Drawing ─────────────────────────────────────────────────────────
    def on_drag(self, event):
        cx = event.x // CELL
        cy = event.y // CELL
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID and 0 <= ny < GRID:
                    dist = abs(dx) + abs(dy)
                    strength = [255, 180, 80, 0, 0][dist] if dist < 4 else 0
                    self.pixels[ny, nx] = min(255, self.pixels[ny, nx] + strength)
        self._redraw()

    def on_release(self, event):
        if self.pixels.sum() > 0:
            self._run_inference()

    def _redraw(self):
        self.canvas.delete("all")
        for y in range(GRID):
            for x in range(GRID):
                v = int(self.pixels[y, x])
                if v > 0:
                    g = f"#{v:02x}{v:02x}{v:02x}"
                    self.canvas.create_rectangle(
                        x * CELL, y * CELL,
                        (x + 1) * CELL, (y + 1) * CELL,
                        fill=g, outline=""
                    )

    # ── Inference ────────────────────────────────────────────────────────
    def _run_inference(self):
        probs = forward(self.pixels)
        digit = int(np.argmax(probs))
        conf  = probs[digit] * 100
        self.pred_var.set(f"Prediction: {digit}   ({conf:.1f}%)")
        self._update_bars(probs, digit)

    def _update_bars(self, probs, best):
        for i, (bc, p) in enumerate(zip(self.bar_cvs, probs)):
            bc.delete("all")
            w = int(p * (CANVAS_SIZE - 30))
            color = "#4CAF50" if i == best else "#1976D2"
            if w > 0:
                bc.create_rectangle(0, 1, w, 17, fill=color, outline="")
            bc.create_text(
                max(w + 4, 4), 9,
                text=f"{p*100:.1f}%",
                anchor="w",
                fill="#ccc",
                font=("Helvetica", 9)
            )

    def clear(self):
        self.pixels[:] = 0
        self.canvas.delete("all")
        self.pred_var.set("Draw a digit above")
        for bc in self.bar_cvs:
            bc.delete("all")


root = tk.Tk()
App(root)
root.mainloop()
