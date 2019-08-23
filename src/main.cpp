#ifndef GLEW_STATIC
	#define GLEW_STATIC
#endif
#include <GL/glew.h> // Take care: GLEW should be included before GLFW
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "libs/helper_cuda.h"
#include "libs/helper_cuda_gl.h"

#include <string>
#include <filesystem>
#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"
#include "gl_tools.h"
#include "glfw_tools.h"

using namespace std;

// GLFW
GLFWwindow* g_window;
const int WIDTH = 1024;
const int HEIGHT = 1024;

// OpenGL
GLuint VBO, VAO, EBO;
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

// Cuda <-> OpenGl interop resources
void* cuda_dev_render_buffer; // Cuda buffer for initial render
struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result
extern "C" void
// Forward declaration of CUDA render
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

// CUDA
size_t size_tex_data;
unsigned int num_texels;
unsigned int num_values;

#define FULLSCREENTRIANGLE 1
#if FULLSCREENTRIANGLE
const char* const glsl_drawtex_vertshader_src = R"(
#version 460 core
vec3 position = vec3(vec2(gl_VertexID % 2, gl_VertexID / 2) * 4.0 - 1, 0);
vec2 texCoord = (position.xy + 1) * 0.5;
)"
#else
const char* const glsl_drawtex_vertshader_src = R"(
#version 460 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;
)"
#endif
R"(
layout(location = 0) out vec2 ourTexCoord;

void main()
{
	gl_Position = vec4(position, 1.0f);
	ourTexCoord = texCoord;
}
)";

const char* const glsl_drawtex_fragshader_src = R"(
#version 460 core

layout(location = 0) in vec2 ourTexCoord;
layout(location = 0) out vec4 color;

layout(location = 0) uniform usampler2D tex;

void main()
{
	vec4 c = texture(tex, ourTexCoord);
	color = c / 255.0;
}
)";

// QUAD GEOMETRY
const GLfloat vertices[] = {
	// Positions       | Texture Coords
	 1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // Top Right
	 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // Bottom Right
	-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // Bottom Left
	-1.0f,  1.0f, 0.0f, 0.0f, 1.0f  // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
const GLuint indices[] = {  // Note that we start from 0!
	0, 1, 3,  // First Triangle
	1, 2, 3   // Second Triangle
};


// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
static void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y)
{
	// create an OpenGL texture
	glGenTextures(1, gl_tex);
	glBindTexture(GL_TEXTURE_2D, *gl_tex);
	glObjectLabel(GL_TEXTURE, *gl_tex, sizeof("output"), "output");
	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, size_x, size_y);
	// Register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

static void initGLBuffers()
{
	// create texture that will receive the result of cuda kernel
	createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, WIDTH, HEIGHT);
	// create shader program
	drawtex_v = GLSLShader("fullscreen quad vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
	drawtex_f = GLSLShader("fullscreen quad fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
	shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
	shdrawtex.compile();
}

// Keyboard
static void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	switch (key)
	{
	case GLFW_KEY_ESCAPE:
		glfwSetWindowShouldClose(window, GL_TRUE);
		break;
	}
}

static void APIENTRY glCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
	GLsizei length, const GLchar* msg, const void* data)
{
	const char* sourceStr = "UNKNOWN";
	const char* typeStr = "UNKNOWN";
	const char* severityStr = "UNKNOWN";

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:
		sourceStr = "API";
		break;

	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
		sourceStr = "WINDOW SYSTEM";
		break;

	case GL_DEBUG_SOURCE_SHADER_COMPILER:
		sourceStr = "SHADER COMPILER";
		break;

	case GL_DEBUG_SOURCE_THIRD_PARTY:
		sourceStr = "THIRD PARTY";
		break;

	case GL_DEBUG_SOURCE_APPLICATION:
		sourceStr = "APPLICATION";
		break;
	}

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:
		typeStr = "ERROR";
		break;

	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		typeStr = "DEPRECATED";
		break;

	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		typeStr = "UNDEFINED";
		break;

	case GL_DEBUG_TYPE_PORTABILITY:
		typeStr = "PORTABILITY";
		break;

	case GL_DEBUG_TYPE_PERFORMANCE:
		typeStr = "PERFORMANCE";
		break;

	case GL_DEBUG_TYPE_OTHER:
		typeStr = "OTHER";
		break;

	case GL_DEBUG_TYPE_MARKER:
		typeStr = "MARKER";
		break;
	}

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:
		severityStr = "HIGH";
		break;

	case GL_DEBUG_SEVERITY_MEDIUM:
		severityStr = "MEDIUM";
		break;

	case GL_DEBUG_SEVERITY_LOW:
		severityStr = "LOW";
		break;

	case GL_DEBUG_SEVERITY_NOTIFICATION:
		severityStr = "NOTIFICATION";
		break;
	}

	printf("[%s] %s: [%s] %d: %s\n",
		severityStr, typeStr, sourceStr, id, msg);
	if (type == GL_DEBUG_TYPE_ERROR || severity == GL_DEBUG_SEVERITY_HIGH)
		glfwSetWindowShouldClose(g_window, true);
}

static bool initGL()
{
	glewExperimental = GL_TRUE; // need this to enforce core profile
	GLenum err = glewInit();
	glGetError(); // parse first error
	if (err != GLEW_OK) {// Problem: glewInit failed, something is seriously wrong.
		printf("glewInit failed: %s /n", glewGetErrorString(err));
		exit(1);
	}
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(glCallback, nullptr);
	glViewport(0, 0, WIDTH, HEIGHT); // viewport for x,y to normalized device coordinates transformation
	return true;
}

static void initCUDABuffers()
{
	// set up vertex data parameters
	num_texels = WIDTH * HEIGHT;
	num_values = num_texels * 4;
	size_tex_data = sizeof(GLuint) * num_values;
	// We don't want to use cudaMallocManaged here - since we definitely want
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output
}

static bool initGLFW()
{
	if (!glfwInit())
		exit(EXIT_FAILURE);
	// These hints switch the OpenGL profile to core
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	g_window = glfwCreateWindow(WIDTH, WIDTH, " CUDA Buddhabrot", nullptr, nullptr);
	if (!g_window) { glfwTerminate(); exit(EXIT_FAILURE); }
	glfwMakeContextCurrent(g_window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(g_window, keyboardfunc);
	return true;
}

static void generateCUDAImage()
{
	// calculate grid size
	dim3 block(16, 16, 1);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1); // 2D grid, every thread will compute a pixel
	launch_cudaRender(grid, block, 0, (unsigned int *) cuda_dev_render_buffer, WIDTH); // launch with 0 additional shared memory allocated

	// We want to copy cuda_dev_render_buffer data to the texture
	// Map buffer objects to get CUDA device pointers
	cudaArray* texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
}

static void display()
{
	generateCUDAImage();
	glfwPollEvents();
	// Clear the color buffer
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

	shdrawtex.use();

	glBindVertexArray(VAO); // binding VAO automatically binds EBO
#if FULLSCREENTRIANGLE
	// dummy VAO still required in core profile
	glDrawArrays(GL_TRIANGLES, 0, 3);
#else
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
#endif
	glBindVertexArray(0); // unbind VAO

	// Swap the screen buffers
	glfwSwapBuffers(g_window);
}

int main(int argc, char* argv[])
{
	initGLFW();
	initGL();

	printGLFWInfo(g_window);
	printGlewInfo();
	printGLInfo();

	findCudaGLDevice(argc, (const char**)argv);
	initGLBuffers();
	initCUDABuffers();

	// Generate buffers
	glGenVertexArrays(1, &VAO);

	// Buffer setup
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO); // all next calls wil use this VAO (descriptor for VBO)

#if !FULLSCREENTRIANGLE
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Texture attribute (2 floats)
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
#endif
	glBindVertexArray(0);

	// Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
	// A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
	// This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.

	while (!glfwWindowShouldClose(g_window))
	{
		display();
		glfwPollEvents();
	}

	glfwDestroyWindow(g_window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}