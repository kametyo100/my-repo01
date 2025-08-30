import tkinter as tk
import tkinter.messagebox as messagebox

message = 'Hello, World!'
print(message)

root = tk.Tk()
root.withdraw()  # メインウィンドウを表示しないようにする
messagebox.showinfo("タイトル", "こんにちは")
