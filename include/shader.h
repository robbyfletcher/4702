// -*- c++ -*-

#ifndef SHADER_H
#define SHADER_H

#include <stdio.h>
#include <string>
#include "pstring.h"
#include "util.h"

static uint pobject_in_use = 0;

class pShader {
public:
  pShader(const char *path){ init_vgf_or_c(path,NULL); }
  pShader(const char *path, const char *main_body_vs,
          const char *main_body_fs = NULL)
  {init(path,main_body_vs,main_body_fs);}
  pShader(const char *path, const char *main_body_vs,
          const char *main_body_gs, const char *main_body_fs)
  {init_vgf_or_c(path,main_body_vs,main_body_gs,main_body_fs);}
  pShader()
  {
    validated = true;
    pobject = 0;
  }
# ifdef GL_VERTEX_SHADER
  GLint uniform_location(const char *name)
  {
    const GLint rv = ptr_glGetUniformLocation(pobject,name);
    if ( rv == -1 ) pError_Msg("Could not get uniform \"%s\".",name);
    return rv;
  }
  // Note: Location is "generic attribute index"
  GLint attribute_location(const char *name)
  {return glGetAttribLocation(pobject,name);}
  GLint varying_location(const char *name)
  {return ptr_glGetVaryingLocationNV(pobject, name);}

private:

  GLuint shader_load
  (GLenum shader_type, const char *source_path, const char *main_body)
  {
    const GLuint sobject = glCreateShader(shader_type);
    pError_Check();
    if ( !sobject )
      {
        // Could not create shader, perhaps feature not supported.
        return -1;
      }

    FILE* const shader_h = fopen(source_path,"r");
    if ( !shader_h )
      {
        fprintf(stderr,"Shader source file %s could not be open.\n",
                source_path);
        return -1;
      }
    std::string shader_define;

    if ( main_body )
      {
        const char* const stype_str =
          shader_type == GL_VERTEX_SHADER ? "VERTEX_SHADER" :
          shader_type == GL_FRAGMENT_SHADER ? "FRAGMENT_SHADER" :
          shader_type == GL_GEOMETRY_SHADER ? "GEOMETRY_SHADER" :
          "unknown_shader";
        shader_define += "#define _";
        shader_define += stype_str;
        shader_define += "_\n";
      }

    std::string file_text;
    while ( !feof(shader_h) )
      {
        const int c = getc(shader_h);
        if ( c == EOF ) break;
        file_text += c;
      }
    fclose(shader_h);

    std::string shader_text;

    // Move the #version directive to the first line.
    //
    const size_t v_start = file_text.find("#version");
    if ( v_start != std::string::npos )
      {
        const size_t v_end = file_text.find("\n",v_start) + 1;
        shader_text = file_text.substr(v_start,v_end-v_start)
          + shader_define
          + file_text.substr(0,v_start)
          + file_text.substr(v_end);
      }
    else
      shader_text = shader_define + file_text;

    if ( main_body )
      {
        shader_text += "void main() {\n";
        shader_text += main_body;
        shader_text += "}\n";
      }

    const char *shader_text_lines = shader_text.c_str();
    if ( 0 )
      printf("\nThe re-arranged shader code:\n%s\n", shader_text_lines);

    glShaderSource(sobject,1,&shader_text_lines,NULL);
    glCompileShader(sobject);
    int rv;
    glGetShaderiv(sobject,GL_COMPILE_STATUS,&rv);
    int info_log_length;
    if ( !rv )
      {
        printf(" Compile status for %s: %d\n",main_body,rv);
        glGetShaderiv(sobject,GL_INFO_LOG_LENGTH,&info_log_length);
        char* const info_log = (char*) alloca(info_log_length+1);
        glGetShaderInfoLog(sobject,info_log_length+1,NULL,info_log);
        printf(" Info log:\n%s\n",info_log);
        return -1;
      }

    glAttachShader(pobject,sobject);
    pError_Check();
    return sobject;
  }

public:
  GLuint init
  (const char *source_pathp,
   const char *main_body_vs, const char *main_body_fs = NULL)
  {
    return init_vgf_or_c(source_pathp,main_body_vs,NULL,main_body_fs);
  }

  GLuint init_vgf_or_c
  (const char *source_pathp,
   const char *main_body_vs,
   const char *main_body_gs = NULL,
   const char *main_body_fs = NULL)
  {
    ready = false;
    validated = false;
    source_path = source_pathp;

    const bool compute_shader = !main_body_vs && !main_body_gs && !main_body_fs;

    pobject = glCreateProgram();
    if ( pobject == 0 ) return 0;

    cs_object = compute_shader 
      ? shader_load(GL_COMPUTE_SHADER,source_path,NULL) : 0;
    if ( cs_object < 0 ) return 0;

    vs_object = main_body_vs 
      ? shader_load(GL_VERTEX_SHADER,source_path,main_body_vs) : 0;
    if ( vs_object < 0 ) return 0;

    gs_object = main_body_gs 
      ? shader_load(GL_GEOMETRY_SHADER,source_path,main_body_gs) : 0;
    if ( gs_object < 0 ) return 0;


    fs_object = main_body_fs
      ? shader_load(GL_FRAGMENT_SHADER,source_path,main_body_fs) : 0;
    if ( fs_object < 0 ) return 0;

    pError_Check();
    glLinkProgram(pobject);
    pError_Check();

    GLint link_status;
    glGetProgramiv(pobject,GL_LINK_STATUS,&link_status);
    if ( !link_status )
      {
        printf(" Link status for %s %d\n",
               source_path.s, link_status);
        GLint info_log_length;
        glGetProgramiv(pobject,GL_INFO_LOG_LENGTH,&info_log_length);
        char* const prog_info_log = (char*) alloca(info_log_length+1);
        glGetProgramInfoLog(pobject,info_log_length+1,NULL,prog_info_log);
        printf(" Program Info log:\n%s\n",prog_info_log);
        return 0;
      }

    ready = true;
    return pobject;
  }

  void print_active_attrib()
  {
    if ( !pobject ) return;
    int active_attributes, max_length;
    pError_Check();
    glGetProgramiv(pobject,GL_ACTIVE_ATTRIBUTES,&active_attributes);
    pError_Check();
    glGetProgramiv(pobject,GL_ACTIVE_ATTRIBUTE_MAX_LENGTH,&max_length);
    pError_Check();
    char* const buffer = (char*) alloca(max_length);
    for ( int i=0; i<active_attributes; i++ )
      {
        int size;
        GLenum type;
        glGetActiveAttrib(pobject,i,max_length,NULL,&size,&type,buffer);
        pError_Check();
        const int location = attribute_location(buffer);
        printf("A %2d: SIZE %2d  TYPE %6d  %s\n",location,size,type,buffer);
      }
  }
  void print_active_uniform()
  {
    if ( !pobject ) return;
    int active_attributes, max_length;
    glGetProgramiv(pobject,GL_ACTIVE_UNIFORMS,&active_attributes);
    glGetProgramiv(pobject,GL_ACTIVE_UNIFORM_MAX_LENGTH,&max_length);
    char* const buffer = (char*) alloca(max_length);
    for ( int i=0; i<active_attributes; i++ )
      {
        int size;
        GLenum type;
        glGetActiveUniform(pobject,i,max_length,NULL,&size,&type,buffer);
        printf("U %2d: SIZE %2d  TYPE %6d  %s\n",i,size,type,buffer);
      }
  }
  void print_active_varying()
  {
#ifndef GL_NV_transform_feedback
    return;
#else
    if ( !pobject ) return;
    int active_attributes, max_length;
    glGetProgramiv(pobject,GL_ACTIVE_VARYINGS_NV,&active_attributes);
    glGetProgramiv(pobject,GL_ACTIVE_VARYING_MAX_LENGTH_NV,&max_length);
    char* const buffer = (char*) alloca(max_length);
    for ( int i=0; i<active_attributes; i++ )
      {
        int size;
        GLenum type;
        ptr_glGetActiveVaryingNV(pobject,i,max_length,NULL,&size,&type,buffer);
        printf("V %2d: SIZE %2d  TYPE %6d  %s\n",i,size,type,buffer);
      }
#endif
  }
  void validate_once()
  {
    if ( validated ) return;
    validate();
  }
  void validate()
  {
    validated = true;
    glValidateProgram(pobject);
    int validate_status;
    glGetProgramiv(pobject,GL_VALIDATE_STATUS,&validate_status);
    printf(" Validate status for %s: %d\n",source_path.s,validate_status);
    int info_log_length;
    glGetProgramiv(pobject,GL_INFO_LOG_LENGTH,&info_log_length);
    char* const prog_info_log = (char*) alloca(info_log_length+1);
    glGetProgramInfoLog(pobject,info_log_length+1,NULL,prog_info_log);
    printf(" Validation log:\n%s\n",prog_info_log);
  }

  bool use()
  {
    if ( pobject_in_use == pobject ) return false;
    pError_Check();
    glUseProgram(pobject);
    pError_Check();
    pobject_in_use = pobject;
    return true;
  }

#else
  // OpenGL Before Vertex Shaders

  GLint uniform_location(const char *name) { return 0; }
  GLint attribute_location(const char *name) { return 0; }
  GLint varying_location(const char *name) { return 0; }
  uint init(const char *source_pathp, const char *main_body)
  {
    validated = false;
    source_path = source_pathp;
    pobject = 0;
    pobject_in_use = 0;
    return pobject;
  }
  void print_active_attrib() { }
  void print_active_uniform() { }
  void print_active_varying() { }
  void validate_once() { }
  void validate() { };
  bool use() { return false; }

#endif

  bool okay(){ return ready; }

  pString source_path;
  GLuint pobject;
  GLuint vs_object, gs_object, fs_object, cs_object;
  bool validated;
private:
  bool ready;
};



#endif
