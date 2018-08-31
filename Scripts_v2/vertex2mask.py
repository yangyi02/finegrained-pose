"""
    Show the usage of the 3D pose annotation by visualizing a sample annotation.
    The visualization is done by laying the projected 3D model
    onto the 2D image. By default, this script visualize the first training
    image of StanfordCars 3D dataset.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from OpenGL import GL
from OpenGL.GL import *
import cyglfw3 as glfw
import glutils
import scipy
from scipy.io import loadmat


def generate_binary_mask(faces, vertices_2d, width, height):
    if not glfw.Init():
        print('glfw not initialized')
        sys.exit()

    version = 3, 3
    glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, version[0])
    glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, version[1])
    glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, 1)
    glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.WindowHint(glfw.VISIBLE, 1)

    window = glfw.CreateWindow(width, height, 'Quad')
    if not window:
        print('glfw window not created')
        glfw.Terminate()
        sys.exit()

    glfw.MakeContextCurrent(window)

    strVS = """
        #version 330
        layout(location = 0) in vec2 aPosition;

        void main() {
            gl_Position = vec4(vec3(aPosition, 0), 1.0);
        }
        """
    strFS = """
        #version 330
        out vec3 fragColor;

        void main() {
            fragColor = vec3(0, 1, 0);
        }
        """

    program = glutils.loadShaders(strVS, strFS)
    glUseProgram(program)

    element_array = np.reshape(faces, -1)
    elementData = np.array(element_array, np.uint32)

    vertex_array = np.reshape(vertices_2d, -1)
    vertexData = np.array(vertex_array, np.float32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * len(elementData), elementData, GL_STATIC_DRAW)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    glBindVertexArray(0)

    GL.glClearColor(0, 0, 0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    glUseProgram(program)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(element_array), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    pixel_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)

    im = np.array(pixel_data)
    mask = Image.frombuffer('RGB', (width, height), im, 'raw', 'RGB')
    mask = np.array(mask)
    glfw.Terminate()
    return mask


def visualize_binary_mask(im, mask):
    """
        Visualize a single sample with segmentation mask
    """
    mask = np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))  # Make mask a green mask
    img_array = im.astype(np.float) / 255.0 * 0.8 + mask.astype(np.float) / 255.0 * 0.2
    plt.subplots()
    plt.imshow(img_array)
    plt.axis('off')
    plt.show()
    plt.subplots()
    plt.imshow(mask)
    plt.axis('off')
    plt.show()


def add_image_binary_mask(im, mask):
    mask = np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))  # Make mask a green mask
    img_array = im.astype(np.float) / 255.0 * 0.8 + mask.astype(np.float) / 255.0 * 0.2
    return img_array


def main():
    """
        Visualize the first sample of a set of annotations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        default='./aeroplane'
    )
    args = parser.parse_args()

    # load annotation
    files = os.listdir(args.input_dir)

    for file in files:
        if file.endswith(".JPEG"):
            img_name = os.path.join(args.input_dir, file)
            print("Processing %s" % img_name)
            im = np.array(Image.open(img_name))
            file_name, ext = os.path.splitext(img_name)
            mat_name = file_name + '.mat'
            anno = loadmat(mat_name)
            faces = anno['face']
            vertices_2d = anno['x2d']

            h, w = im.shape[0], im.shape[1]
            vertices_2d[:, 0] = vertices_2d[:, 0] / w * 2 - 1
            vertices_2d[:, 1] = vertices_2d[:, 1] / h * 2 - 1
            mask = generate_binary_mask(faces, vertices_2d, w, h)
            mask = mask[:, :, 1]
            # visualize_binary_mask(im, mask)

            if np.max(mask) < 1.01:
                mask = mask * 255
            segment_file = os.path.join('mask', file_name + '_segment.png')
            segment = np.dstack((mask, mask, mask))
            scipy.misc.imsave(segment_file, segment)
            img = add_image_binary_mask(im, mask)
            image_file = os.path.join('mask', file_name + '_withmask.png')
            scipy.misc.imsave(image_file, img)
            # break


if __name__ == '__main__':
    main()
