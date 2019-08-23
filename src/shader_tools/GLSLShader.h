#pragma once
#include "shader_tools_common.h"

class GLSLShader
{
public:
	GLuint shader = 0;
	GLint compiled = 0;
	GLenum shadertype = 0;
	std::string shader_name;
private:
	std::string shader_src; // internal string representation of shader

public:
	GLSLShader() = default;
	GLSLShader(const std::string &shader_name, const char *shader_text, GLenum shadertype);
	GLSLShader(const std::string &shader_name, const std::string &shader_text, GLenum shadertype);
	std::string getSrc() const; 
	void setSrc(const std::string &new_source); 
	void setSrc(const char* new_source);
	void compile();

private:
	static void getCompilationError(GLuint shader);
};