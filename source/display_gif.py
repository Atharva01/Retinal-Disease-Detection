from PIL import Image
from PIL import ImageTk
import tkinter as tk

# Create a function to display the GIF
def display_gif(gif_path):
    root = tk.Tk()
    root.title("GIF Display")

    # Open the GIF file using Pillow
    image = Image.open(gif_path)

    # Convert the Pillow image to a Tkinter-compatible format
    tk_image = ImageTk.PhotoImage(image)

    # Create a label widget to display the GIF
    label = tk.Label(root, image=tk_image)
    label.pack()

    # Start the Tkinter main loop
    root.mainloop()

# Provide the path to your GIF file
gif_path = "Retinal-Disease-Detection/datasets_drive/training/1st_manual/21_manual1.gif"

# Call the display_gif function with your GIF file path
display_gif(gif_path)
