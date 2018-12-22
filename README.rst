shadertoy-render
================

fork from https://github.com/alexjc/shadertoy-render

**what is it**: creating video from shaders on Shadertoy

**changes**:

1. added texture support, textures in **<0.png to 3.png> bind to iTexture<0-3>**
2. added FrameBuffers, same to Buffer<A-D> on Shadertoy, **file name Buf<0-3>.glsl**, bind to iChannel<0-3>
3. added encoding to \*.mov (frames without compression), \*.webm (v8 codec), both format **support RGBA** (video with alpha), *to edit ffmpeg options* (bitrate/etc) edit option you need on line 808+ shadertoy-render.py
4. fixed iTime(start from 0 on shader launch) and other

**Warning**
many shaders(even top rated) on Shadertoy use clamp/pow/(/0)/...etc, that work in not same way in OpenGL and webbrowser Angle/GLES, 

**example**

each shader(buffer) use static bindings iChannel0->Buf0.glsl, iChannel1->Buf1.glsl, iChannel2->Buf2.glsl, iChannel3->Buf3.glsl

if you need "change" order for shader, use in .glsl file (example set BufA as BufC)

	#define iChannel0 iChannel3
	
use same way to bind iTexture<0-3> as iChannel<0-3> *#define iChannel0 iTexture0*

**check example folder**, command to encode example:

	> python shadertoy-render.py ...

**example 1** video recorded with RGBA and bufA

**example 2** use BufA-D(0-3) and texture
