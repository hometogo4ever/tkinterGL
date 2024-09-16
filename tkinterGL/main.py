from render import *
import warnings
import torch
import os

# Ignore warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the name of the object and the folder path
name = 'Pinkie'
folder_path = 'pony_obj'

if __name__=='__main__':
    # Create a screen and a camera : bind object to screen, bind screen to camera
    screen = Screen(800, 800)
    screen.readObj(folder_path + f'/{name}.obj')    
    camera = Camera([10, 5, 10], [0,0,0], [0,-1,0], 5)
    camera.bindScreen(screen)
    
    # Bind texture to the object
    file_count = len(os.listdir(folder_path + f'/{name}_texture'))
    for i in range(file_count):
        filename = folder_path + f'/{name}_texture/{name}_{i}.jpg'
        camera.screen.bindTexture(filename)

    # Render the object
    camera.render()
    camera.screen.tk.mainloop()