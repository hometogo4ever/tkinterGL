"""
Description : This file contains the function to read the obj file and return the vertices, faces, textures and normals of the object.

functions:
    readObjFile(objectname, type) : This function reads the obj file and returns the vertices, faces, textures and normals of the object.
    It takes the following parameters:
        objectname : str : The name of the obj file.
        type : int : The type of the object. If type is 1, only vertices are returned. If type is 2, vertices, textures and normals are returned.
    It returns the following:
        vertices : list : The vertices of the object.
        faces : list : The faces of the object.
        textures : list : The textures of the object.
        normals : list : The normals of the object.
        type : int : The type of the object.
"""

def readObjFile(objectname, type):
    # Reading the obj file
    with open(objectname, 'r') as file: 
        lines = file.readlines()

    vertices = []
    textures = []
    faces = []
    normals = []
    texturenum = 0

    # Parsing the obj file
    for line in lines:
        if line[0:6] == 'o obj2': # Skip multiple objects file
            break
        if line[0:6] == 'usemtl': # For multiple textures
            texturenum += 1
        if line[0:2] == 'v ': # Vertices
            vertices.append(tuple(map(float, line[2:].split())))
        elif line[0:2] == 'vt': # Textures Coordinates
            textures.append(tuple(map(float, line[3:].split())))  
        elif line[0:2] == 'vn': # Normals Coordinates
            normals.append(tuple(map(float, line[3:].split())))
        elif line[0:2] == 'f ': # Faces
            if type == 1: # Only Vertices
                faces.append(tuple(map(int, line[2:].split())))
            else: # Vertices and Textures and (Normals)
                face = []
                lst = line[2:].split()
                if len(lst) == 3: # Triangle
                    for i in range(3):
                        vertex = tuple(map(int, lst[i].split('/')))
                        face.append(vertex)
                    faces.append(tuple(face) + (texturenum,))
                elif len(lst) == 4: # Quad : Convert to 2 triangles
                    for i in range(2):
                        vertex1 = tuple(map(int, lst[0].split('/')))
                        vertex2 = tuple(map(int, lst[i+1].split('/')))
                        vertex3 = tuple(map(int, lst[i+2].split('/')))
                        faces.append((vertex1, vertex2, vertex3, texturenum))
    return vertices, faces, textures, normals, type # Return Vertices, Faces, Textures, Normals and Type of the object
