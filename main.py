import tkinter as tk
import tkinter.messagebox as messagebox

message = 'Hello, Koyuki!'
print(message)

root = tk.Tk()
root.withdraw()  # メインウィンドウを表示しないようにする
messagebox.showinfo("タイトル", "お利口こゆきちゃん")
