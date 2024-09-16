"""
Description: This file contains the code for rendering 3D objects on the screen.

classes:
    Screen : This class is used to create a screen and render 3D objects on it.
    Camera : This class is used to create a camera and bind it to the screen.
"""

import torch
import param
from objreader import readObjFile
import tkinter as tk
import math
from PIL import Image, ImageDraw
import time
import threading

"""
vertex2world
This function converts the vertices from the world coordinate to camera coordinate.
Use the camera position, world direction and up vector to calculate the camera coordinate.
Use the rotation matrix to convert the world coordinate to camera coordinate.

Parameters:
    v (n x 3 tensor) : The vertices in the world coordinate. n is the number of vertices.
    camera (Camera) : The camera object. We need direction, Vup and camera position to calculate the camera coordinate.
Returns:
    n x 3 tensor : The vertices in the camera coordinate.

"""
def vertex2world(v, camera):
    v = torch.cat((v, torch.ones(v.size(0), 1, device=param.dev)), dim=1) # Convert 3D vertex to 'point' affine form (v, 1) 
    zvc = torch.tensor(camera.wdir - camera.camera_position, device=param.dev) / torch.norm((camera.wdir - camera.camera_position).float()) # Calculate the z-axis of the camera coordinate
    xvc = torch.cross(camera.vup.float(), zvc) / torch.norm(torch.cross(camera.vup.float(), zvc)) # Calculate the x-axis of the camera coordinate
    yvc = torch.cross(zvc, xvc) # Calculate the y-axis of the camera coordinate
    r = torch.stack((xvc, yvc, zvc)).t() # Create the rotation matrix

    matrixV2W = torch.eye(4, device=param.dev) # Create the view matrix : camera coordinate to world coordinate
    matrixV2W[:3, :3] = r
    matrixV2W[:3, 3] = camera.camera_position
    matrixV2W[3, 3] = 1
    matrixW2V = torch.inverse(matrixV2W) # Create the view matrix : world coordinate to camera coordinate

    return torch.mm(matrixW2V, v.t()).t()[:, :3] # Convert the world coordinate to camera coordinate


"""
world2proj
This function converts the vertices from the camera coordinate to projection coordinate.
Use the projection matrix to convert the camera coordinate to projection coordinate.

Parameters:
    vc (n x 3 tensor) : The vertices in the camera coordinate. n is the number of vertices.
    camera (Camera) : The camera object. We need the screen distance to calculate the projection coordinate.
Returns:
    n x 3 tensor : The vertices in the projection coordinate.
"""
def world2proj(vc, camera):
    projectionMatrix = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,-1/camera.screendistance, 0]], device=param.dev) # Create the projection matrix
    vc = torch.cat((vc, torch.ones(vc.size(0), 1, device=param.dev)), dim=1) # Convert 3D vertex to 'point' affine form (v, 1)
    vc = torch.mm(projectionMatrix, vc.t()) / vc[:, 3] # Convert the camera coordinate to projection coordinate
    vc = vc.t()[:, :3] # Remove the last column
    return vc

"""
proj2norm
This function converts the vertices from the projection coordinate to normalized coordinate.
Use the min and max values of the vertices to normalize the vertices.

Parameters:
    vc (n x 3 tensor) : The vertices in the projection coordinate. n is the number of vertices.
Returns:
    n x 3 tensor : The vertices in the normalized coordinate.
"""
def proj2norm(vc):
    max_values, min_values = param.find_min_max(vc) # Find the min and max values of the vertices
    normalMatrix = torch.eye(4, device=param.dev) # Create the normalization matrix
    normalMatrix[0, 0] = 2 / (max_values[0] - min_values[0]) 
    normalMatrix[1, 1] = 2 / (max_values[1] - min_values[1])
    normalMatrix[2, 2] = 2 / (max_values[2] - min_values[2])
    normalMatrix[0, 3] = - (max_values[0] + min_values[0]) / (max_values[0] - min_values[0])
    normalMatrix[1, 3] = - (max_values[1] + min_values[1]) / (max_values[1] - min_values[1])
    normalMatrix[2, 3] = - (max_values[2] + min_values[2]) / (max_values[2] - min_values[2])
    normalMatrix[3, 3] = 1
    vc = torch.cat((vc, torch.ones(vc.size(0), 1, device=param.dev)), dim=1) # Convert 3D vertex to 'point' affine form (v, 1)
    return torch.mm(normalMatrix, vc.t()).t()[:, :3] # Convert the projection coordinate to normalized coordinate

"""
norm2screen
This function converts the vertices from the normalized coordinate to screen coordinate.
Use the screen matrix to convert the normalized coordinate to screen coordinate.

Parameters:
    vc (n x 3 tensor) : The vertices in the normalized coordinate. n is the number of vertices.
    width (int) : The width of the screen.
    height (int) : The height of the screen.
Returns:
    n x 3 tensor : The vertices in the screen coordinate.
"""
def norm2screen(vc, width, height):
    screenMatrix = torch.eye(4, device=param.dev) # Create the screen matrix
    screenMatrix[0, 0] = width / 2
    screenMatrix[1, 1] = height / 2
    screenMatrix[0, 3] = width / 2
    screenMatrix[1, 3] = height / 2
    screenMatrix[2, 2] = 1
    screenMatrix[3, 3] = 1
    vc = torch.cat((vc, torch.ones(vc.size(0), 1, device=param.dev)), dim=1) # Convert 3D vertex to 'point' affine form (v, 1)
    return torch.mm(screenMatrix, vc.t()).t()[:, :3] # Convert the normalized coordinate to screen coordinate

"""
vertex2screen
This function converts the vertices from the world coordinate to screen coordinate.
Use the vertex2world, world2proj, proj2norm and norm2screen functions to convert the vertices.
"""
def vertex2screen(v, width, height, camera):
    return norm2screen(proj2norm(world2proj(vertex2world(v, camera), camera)), width, height)

"""
facenormal
This function calculates the normal of the face.
Use normal of vertex, and face's normal index to calculate the face's normal.
"""
def facenormal(faceindex, normals):
    return torch.mean(normals[faceindex], dim=1)

class Screen:
    """
    Screen class
    This class is used to create a screen and render 3D objects on it.
    For initialization, it takes the width and height of the screen. With that, it creates a tkinter object and a canvas object.
    For rendering, it takes the binding of the camera object and the object to render the object on the screen.
    For shading, it takes the face, normal, light direction, texture, texture file, camera and vertex to calculate the shading of the face.

    Attributes:
        width (int) : The width of the screen.
        height (int) : The height of the screen.
        tk (tkinter.Tk) : The tkinter object.
        canvas (tkinter.Canvas) : The canvas object.
        vertices (n x 3 tensor) : The vertices of the object.
        faces (n x 3 tensor) : The faces of the object.
        normals (n x 3 tensor) : The normals of the object.
        textures (n x 2 tensor) : The textures of the object.
        transformedVertices (n x 3 tensor) : The vertices in the screen coordinate.
        texturefile (list) : The list of texture files.
        texturenum (int) : The number of texture files.

    Methods:
        readObj : This method reads the obj file and sets the vertices, faces, textures and normals of the object.
        bindTexture : This method binds the texture file to the object.
        curling : This method filters the faces based on the direction of the camera.
        shading : This method calculates the shading of the face.
        render : This method renders the object on the screen.
        draw : This method draws the pixel on the screen.
        display : This method displays the screen.

    """
    def __init__(self, width, height):
        # Tkinter object generation
        self.width = width
        self.height = height
        self.tk = tk.Tk()
        self.canvas = tk.Canvas(self.tk, width=width, height=height)

        # Object data
        self.vertices = None
        self.faces = None
        self.normals = None
        self.textures = None

        # Shading, Rendering data
        self.transformedVertices = None
        self.texturefile = []
        self.texturenum = 0

    def readObj(self, filename):
        """
        readObj
        This method reads the obj file and sets the vertices, faces, textures and normals of the object using the readObjFile function.
        Bind the vertices, faces, textures and normals to the object.

        Parameters:
            filename (str) : The name of the obj file.
        Returns:
            None
        """
        print('Loading obj file...')
        vertices, faces, textues, normals, type = readObjFile(filename, 3) # Read the obj file

        # Bind the vertices, faces, textures and normals to the object
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=param.dev) 
        self.textures = torch.tensor(textues, dtype=torch.float32, device=param.dev)
        self.normals = torch.tensor(normals, dtype=torch.float32, device=param.dev)
        self.faces = faces
        print('Loading complete.')

    def bindTexture(self, filename):
        """
        bindTexture
        This method binds the texture file to the object.
        Read the texture file and append it to the texturefile list.(in the form of tensor)

        Parameters:
            filename (str) : The name of the texture file.
        returns:
            None
        """
        print('Loading texture file...')
        image = Image.open(filename) # Read the texture file using PIL
        pixels = list(image.getdata()) # Get the pixel data
        width, height = image.size # Get the width and height of the image

        # Convert the pixel data to tensor and append it to the texturefile list
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)] 
        self.texturefile.append(torch.tensor(pixels, dtype=torch.uint8, device=param.dev))
        self.texturenum += 1

    def curling(self, direction):
        """
        curling
        This method filters the faces based on the direction of the camera.
        Calculate the normal of the face and the angle between the normal and the direction.
        Filter the faces based on the angle : if the angle is greater than 0, then the face is visible.
        Filter out invisible faces and return the filtered faces, type and normal.
        
        Parameters:
            direction (3 x 1 tensor) : The direction of the camera.
        Returns:
            filtered_face (n x 3 tensor) : The filtered faces.
            type (n x 1 tensor) : The type of the faces. (texture file index)
            normal (n x 3 tensor) : The normal of the faces.
        """
        filtered_face = torch.tensor([d[:3] for d in self.faces], device=param.dev, dtype=torch.int32) # Get the face data : without the type
        type = torch.tensor([d[3] for d in self.faces], device=param.dev, dtype=torch.int32) # Get the type of the face
        normalindex = filtered_face[:,:,2] - 1 

        # Calculate the normal of the face, and the angle between the normal and the direction
        normal = torch.mean(self.normals[normalindex], dim=1, dtype=torch.float32)
        direction = direction.unsqueeze(0)  # 1 x 3 텐서로 변환
        angle = torch.matmul(normal, direction.t())
        normal_norm = torch.norm(normal, dim=1, keepdim=True)
        direction_norm = torch.norm(direction)
        angle = angle / (normal_norm * direction_norm)

        # Filter the faces based on the angle
        mask = angle > 0
        mask = mask.squeeze()

        return filtered_face[mask], type[mask], normal[mask] # Return the filtered faces, type and normal

    def shading(self, face, normal, lightdir, texture, texturefile, camera, vertex):
        """
        shading
        This method calculates the shading of the face with Phong Flat Shading.
        First, with texture coordinate and texture file, calculate the RGB values of the face by taking average of the RGB values of the three points.
        Calculate the diffuse color of the face using the angle between the normal and the light direction.
        Calculate the specular color of the face using the angle between the normal and the reflected light direction.
        Calculate the ambient color of the face using the ambient light.
        Calculate the light intensity using the distance between the light and the face.
        Calculate the final color of the face using the diffuse, specular and ambient color.
        Clip the color between 0 and 255 and return the color.
        
        Parameters:
            face (n x 3 tensor) : The face data.
            normal (n x 3 tensor) : The normal of the face.
            lightdir (3 x 1 tensor) : The direction of the light.
            texture (n x 2 tensor) : The texture coordinate of the face.
            texturefile (n x n tensor) : The texture file.
            camera (Camera) : The camera object.
            vertex (n x 3 tensor) : The vertices of the object.
        Returns:
            clipped_light (n x 3 tensor) : The color of the face.
        """
        print('Shading...' + str(face.size(0)) + ' faces') 
        texture_coordinate = texture[face[:,:,1] - 1] # Get the texture coordinate of the face
        vertex_coordinate = vertex[face[:,:,0] - 1] # Get the vertex coordinate of the face
        middlepoint = vertex_coordinate.mean(dim=1) # Calculate the middle point of the face : middle point will be used as light position

        # Convert the texture coordinate to pixel coordinate : round the texture coordinate and clamp it between 0 and texture size
        texture_coordinate[:, :, 0] = torch.round(texture_coordinate[:, :, 0] * texturefile.size(0)).long()
        texture_coordinate[:, :, 1] = torch.round(texture_coordinate[:, :, 1] * texturefile.size(1)).long()
        texture_coordinate[:, :, 0] = torch.clamp(texture_coordinate[:, :, 0], 0, texturefile.size(0) - 1)
        texture_coordinate[:, :, 1] = torch.clamp(texture_coordinate[:, :, 1], 0, texturefile.size(1) - 1)
        texture_coordinate = torch.tensor(texture_coordinate, device=param.dev, dtype=torch.int32)

        # Get the RGB values of the face using the texture coordinate and texture file
        rgb_values = texturefile[texture_coordinate[:,:, 0], texture_coordinate[:,:, 1]] 
        rgb_values = rgb_values.float()
        diffuse_color = rgb_values.mean(dim=1)

        # Calculate the diffuse color of the face using the angle between the normal and the light direction
        lightdir_expanded = lightdir.unsqueeze(0).expand(vertex_coordinate.size(0), -1)
        lightvec = middlepoint - lightdir_expanded
        lightvec = lightvec / torch.norm(lightvec, dim=1, keepdim=True)
        angle = torch.max(torch.sum(normal * lightvec, dim=1, keepdim=True), torch.zeros_like(torch.sum(normal * lightvec, dim=1, keepdim=True)))
        diffuse = angle.expand_as(diffuse_color) * diffuse_color

        # Calculate the specular color of the face using the angle between the normal and the reflected light direction
        viewdir_expanded = camera.camera_position.unsqueeze(0).expand(vertex_coordinate.size(0), -1)
        viewdir = viewdir_expanded - middlepoint
        viewdir = viewdir / torch.norm(viewdir, dim=1, keepdim=True)
        reflectdir = 2 * (normal * torch.sum(normal * lightvec, dim=1, keepdim=True)) - lightvec

        # Calculate the specular color of the face using the angle between the normal and the reflected light direction
        k_s = torch.tensor([0.8, 0.8, 0.8], device=param.dev)
        n = 20
        specular = torch.pow(torch.max(torch.sum(reflectdir * viewdir, dim=1, keepdim=True), torch.zeros_like(torch.sum(reflectdir * viewdir, dim=1, keepdim=True))), n)

        # Calculate the ambient color of the face using the ambient light
        k_a = torch.tensor([0.1, 0.1, 0.1], device=param.dev)
        I_a = 1
        ambient = k_a * I_a

        # Calculate the light intensity using the distance between the light and the face
        I_d = 700
        distance = torch.norm(middlepoint - camera.camera_position.unsqueeze(0), dim=1)
        I_light = I_d / torch.pow(distance, 2)

        # Calculate the final color of the face using the diffuse, specular and ambient color
        specular = specular.expand_as(diffuse_color) * k_s
        light = ambient + I_light.unsqueeze(1) * (diffuse + specular)

        # Clip the color between 0 and 255
        clipped_light = torch.clamp(light * 255, 0, 255)
        clipped_light = clipped_light.squeeze(1)
        return clipped_light
        
    def render(self, camera, zbuffer=True, curling=True):
        """
        render
        This method renders the object on the screen.
        First, convert the vertices from the world coordinate to screen coordinate.
        Filter the faces based on the direction of the camera.
        Calculate the shading of the face using the shading method.
        Z-buffering : Sort the faces based on the z-buffer value.
        Render the faces on the screen using the color of the face with Tkinter Rasterization.
        Pack the canvas and update the screen.

        Parameters:
            camera (Camera) : The camera object.
            zbuffer (bool) : The flag for z-buffering.
            curling (bool) : The flag for curling.
        Returns:
            None
        """
        print('Transforming...')
        self.transformedVertices = vertex2screen(self.vertices, self.width, self.height, camera) # Convert the vertices from the world coordinate to screen coordinate
        
        print('Curling...') 
        direction = camera.camera_position - camera.wdir
        filtered_face, filtered_type, filtered_normal = self.curling(direction) # Filter the faces based on the direction of the camera

        print('Texture mapping...')
        col = []
        for i in range(self.texturenum): # Calculate the shading of the face using the shading method
            # for each texture file, calculate the shading of the face
            texture = self.texturefile[i]
            mask = filtered_type == i+1
            selected_face = filtered_face[mask]
            selected_normal = filtered_normal[mask]
            col.append(self.shading(selected_face, selected_normal, torch.tensor([20, 0, 20], device=param.dev), self.textures, texture, camera, self.transformedVertices))
        color = torch.cat(col, dim=0) # Concatenate the color of the face

        print('Z buffering...')
        faces_vertices = self.transformedVertices[filtered_face[:,:,0] - 1] # Get the vertices of the face
        z_buffer = faces_vertices[:,:,2].mean(dim=1)
        sorted_indices = torch.argsort(z_buffer, descending=True)
        if zbuffer: # Sort the faces based on the z-buffer value
            sorted_face_vertices = faces_vertices[sorted_indices]
            color = color[sorted_indices]
        else: # If z-buffering is not used, render the faces without sorting
            sorted_face_vertices = faces_vertices
            
        print('Rendering...')
        # Render the faces on the screen using the color of the face with Tkinter Rasterization
        t1 = time.time() # Start the timer : this rasterization process takes a long time
        for face, col in zip(sorted_face_vertices, color): 
            x1, y1 = face[0][:2].tolist()
            x2, y2 = face[1][:2].tolist()
            x3, y3 = face[2][:2].tolist()
            r, g, b = col.int().tolist()
            self.canvas.create_polygon(float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), fill=f'#{r:02x}{g:02x}{b:02x}', outline='') # Draw the face on the screen
        print('Rendering complete. Execution Time: ', time.time() - t1)
        self.canvas.update_idletasks() # Update the screen
        self.canvas.pack() # Pack the canvas

class Camera:
    """
    Camera class
    This class is used to create a camera and bind it to the screen.
    For initialization, it takes the position, direction, up vector and screen distance of the camera.
    For binding, it takes the screen object to bind the camera to the screen.
    For rotation, it takes the pitch, yaw and roll to rotate the camera.
    For rendering, it takes the transformed vertices to render the object on the screen.
    For rotation loop, it takes the dt to rotate the camera continuously.
    
    Attributes:
        camera_position (3 x 1 tensor) : The position of the camera.
        wdir (3 x 1 tensor) : The direction of the camera.
        vup (3 x 1 tensor) : The up vector of the camera.
        screendistance (float) : The screen distance of the camera.
        screen (Screen) : The screen object.
    Methods:
        bindScreen : This method binds the screen object to the camera object.
        on_key_press : This method rotates the camera based on the key press event.
        render : This method renders the object on the screen.
        rotate : This method rotates the camera based on the pitch, yaw and roll.
        rotateloop : This method rotates the camera continuously based on the dt.
    """
    def __init__(self, position, dir, up, dist):
        self.camera_position = torch.tensor(position, device=param.dev, dtype=torch.float32)
        self.wdir = torch.tensor(dir, device=param.dev, dtype=torch.float32)
        self.vup = torch.tensor(up, device=param.dev)   
        self.screendistance = dist
        self.screen = None
    def bindScreen(self, screen):
        """
        bindScreen
        This method binds the screen object to the camera object.
        
        Parameters:
            screen (Screen) : The screen object.
        Returns:
            None
        """
        self.screen = screen
        self.screen.tk.bind('<KeyPress>', self.on_key_press)
    def on_key_press(self, event):
        """
        on_key_press
        This method rotates the camera based on the key press event.
        Rotate the camera based on the key press event and delete the screen to render the object on the screen.
        For binding the key press event, use the tkinter bind method.
        """
        if event.keysym == 'q':
            print('Left key pressed')
            self.rotate(0, 0.5, 0)
        elif event.keysym == 'e':
            print('Right key pressed')
            self.rotate(0, -0.5, 0)
        elif event.keysym == 'r':
            print('Up key pressed')
            self.rotate(0.5, 0, 0)
        elif event.keysym == 'f':
            print('Down key pressed')
            self.rotate(-0.5, 0, 0)
        elif event.keysym == 't':
            print('Forward key pressed')
            self.rotate(0, 0, 0.5)
        elif event.keysym == 'g':
            print('Backward key pressed')
            self.rotate(0, 0, -0.5)
        else:
            return
        print('Deleting...')
        self.screen.canvas.delete('all')
        self.render()
    def render(self):
        self.transformedVertices = self.screen.render(self)
    def rotate(self, pitch, yaw, roll):
        """
        rotate
        This method rotates the camera based on the pitch, yaw and roll.
        Use the rotation matrix to rotate the camera based on the pitch, yaw and roll.

        Parameters:
            pitch (float) : The pitch of the camera.
            yaw (float) : The yaw of the camera.
            roll (float) : The roll of the camera.
        Returns:
            None
        """
        rotate_pitch = torch.tensor([
            [1, 0, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch), 0],
            [0, math.sin(pitch), math.cos(pitch), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=param.dev)

        rotate_yaw = torch.tensor([
            [math.cos(yaw), 0, math.sin(yaw), 0],
            [0, 1, 0, 0],
            [-math.sin(yaw), 0, math.cos(yaw), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=param.dev)

        rotate_roll = torch.tensor([
            [math.cos(roll), -math.sin(roll), 0, 0],
            [math.sin(roll), math.cos(roll), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=param.dev)

        # 3x1 벡터를 4x1 벡터로 변환하고 마지막 요소를 1로 채우기
        camera_position_4x1 = torch.cat((self.camera_position, torch.tensor([1], dtype=torch.float32, device=param.dev)))

        # 행렬 곱셈 순서 확인
        matrix = rotate_pitch @ rotate_yaw @ rotate_roll @ camera_position_4x1

        # 결과를 3x1 벡터로 변환
        self.camera_position = matrix[:3]
    def rotateloop(self, dt):
        def loop():
            self.screen.canvas.delete('all')
            self.rotate(0, 0, 0.1)
            self.render()
            self.screen.tk.after(int(dt * 1000), loop)  # Repeat after `dt` milliseconds
        loop()
        self.screen.tk.mainloop()

        