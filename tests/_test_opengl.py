import glfw
import imageio
import imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl

def array_to_texture(array):
    print("{}, {}".format(array.shape, array.dtype))

    # generate texture reference
    tex = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)

    # mipmap level
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexParameteriv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_RGBA, [gl.GL_RED]*4)

    # bind data
    ny, nx = array.shape
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, # texture 
        gl.GL_R8, nx, ny, 0, # texture format
        gl.GL_RED, gl.GL_UNSIGNED_BYTE, array # data format
    )
    
    return tex

def main():
    path = 'lena512.bmp'
    I = imageio.imread(path)
    ny, nx = I.shape

    window = impl_glfw_init(I.shape, path)
    impl = GlfwRenderer(window)

    tex = array_to_texture(I)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        nwx, nwy = glfw.get_window_size(window)

        impl.process_inputs()

        imgui.new_frame()

        imgui.set_next_window_position(0, 0)
        
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.push_style_var(imgui.STYLE_SCROLLBAR_SIZE, 0)

        imgui.begin('', False, imgui.WINDOW_NO_TITLE_BAR|imgui.WINDOW_NO_RESIZE|imgui.WINDOW_NO_MOVE|imgui.WINDOW_NO_COLLAPSE|imgui.WINDOW_ALWAYS_AUTO_RESIZE|imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_SAVED_SETTINGS)
        
        imgui.image(tex, nwx, nwy)

        imgui.end()

        imgui.pop_style_var(4)

        gl.glClearColor(1., 1., 1., 1.)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init(shape, title='untitled'):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    #glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)

    # Create a windowed mode window and its OpenGL context
    ny, nx = shape
    window = glfw.create_window(
        int(nx)+1, int(ny)+1, title, None, None
    )
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

if __name__ == "__main__":
    main()